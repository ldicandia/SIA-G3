//! Run registry, worker pool and readers of the files the training children
//! write. The server never trains in-process: every run is a `tp3-rust
//! exercise{2|3} train` subprocess driven by a generated `config.toml`.

use std::{
    collections::{HashMap, VecDeque},
    ffi::OsString,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Condvar, Mutex, MutexGuard},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::Serialize;

use super::spec::{study_toml, Exercise};
use crate::digits::{config::CandidateConfig, live_dashboard::LiveMetric};

pub const MAX_QUEUED_RUNS: usize = 100;
const STDERR_TAIL_LINES: usize = 20;
const ERROR_MAX_CHARS: usize = 2000;
/// The UDP side of the publisher is intentionally unused: the discard port
/// keeps a user's own `monitor` on 7879 free of UI runs.
const DISCARD_ADDRESS: &str = "127.0.0.1:9";

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Queued,
    Running,
    Done,
    Failed,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RunResult {
    pub epochs_run: usize,
    pub best_epoch: usize,
    pub validation_accuracy: f64,
    pub validation_loss: f64,
    /// `None` when the summary holds NaN (no history row at the best epoch).
    pub train_accuracy_at_best: Option<f64>,
    pub parameter_count: usize,
    pub stopped_early: bool,
    pub seconds: f64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct MetricPoint {
    pub epoch: usize,
    pub train_loss: f64,
    pub validation_loss: Option<f64>,
    pub train_accuracy: f64,
    pub validation_accuracy: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunSummary {
    pub id: String,
    pub group: Option<String>,
    pub exercise: Exercise,
    pub name: String,
    pub axis: String,
    pub status: RunStatus,
    pub error: Option<String>,
    pub last_epoch: Option<usize>,
    pub created_ms: u64,
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
    pub validation_ratio: f64,
    pub split_seed: u64,
    pub candidate: CandidateConfig,
    pub result: Option<RunResult>,
}

#[derive(Clone, Debug)]
pub struct ManagerConfig {
    pub exe: PathBuf,
    pub output_root: PathBuf,
    pub digits: PathBuf,
    pub more_digits: PathBuf,
    pub baseline_model: PathBuf,
    pub max_parallel: usize,
}

/// Sweep membership: runs of one group share an id and a concurrency cap.
#[derive(Clone, Debug)]
pub struct GroupSpec {
    pub id: String,
    pub parallel: usize,
}

#[derive(Clone, Debug)]
pub struct BatchSpec {
    pub exercise: Exercise,
    pub validation_ratio: f64,
    pub split_seed: u64,
    pub candidates: Vec<CandidateConfig>,
    pub group: Option<GroupSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Job {
    pub run_id: String,
    pub group: Option<String>,
    pub parallel_cap: usize,
}

struct State {
    runs: Vec<RunSummary>,
    queue: VecDeque<Job>,
    running_by_group: HashMap<String, usize>,
    next_seq: u64,
    next_group: u64,
}

/// Index of the first queued job that may start: ungrouped, or whose group is
/// below its concurrency cap. A saturated group never blocks other jobs.
pub fn next_runnable(
    queue: &VecDeque<Job>,
    running_by_group: &HashMap<String, usize>,
) -> Option<usize> {
    queue.iter().position(|job| match &job.group {
        None => true,
        Some(group) => running_by_group.get(group).copied().unwrap_or(0) < job.parallel_cap,
    })
}

/// Rayon threads per training when `parallel` trainings share the machine.
pub fn threads_per_run(cores: usize, parallel: usize) -> usize {
    (cores / parallel.max(1)).max(1)
}

struct Shared {
    config: ManagerConfig,
    session_secs: u64,
    state: Mutex<State>,
    wakeup: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }
}

pub struct RunManager {
    shared: Arc<Shared>,
}

pub fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn format_run_id(session_secs: u64, seq: u64) -> String {
    format!("{session_secs}-{seq}")
}

/// The only gate between a URL segment (or query value) and the registry:
/// ASCII digits around a single '-', with an optional leading `g` that marks
/// group ids.
pub fn is_valid_id(id: &str) -> bool {
    let id = id.strip_prefix('g').unwrap_or(id);
    match id.split_once('-') {
        Some((left, right)) => {
            !left.is_empty()
                && !right.is_empty()
                && left.bytes().all(|byte| byte.is_ascii_digit())
                && right.bytes().all(|byte| byte.is_ascii_digit())
        }
        None => false,
    }
}

/// Argument vector of the training child. Every path is its own element and
/// nothing here ever goes through a shell.
pub fn build_train_args(
    exercise: Exercise,
    config: &ManagerConfig,
    run_dir: &Path,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![exercise.as_str().into(), "train".into()];
    let mut push = |flag: &str, value: &Path| {
        args.push(flag.into());
        args.push(value.as_os_str().to_owned());
    };
    match exercise {
        Exercise::Exercise2 => push("--data", &config.digits),
        Exercise::Exercise3 => {
            push("--data", &config.more_digits);
            push("--previous-data", &config.digits);
            push("--baseline-model", &config.baseline_model);
        }
    }
    push("--config", &run_dir.join("config.toml"));
    push("--output", run_dir);
    args.push("--live-publish-only".into());
    args.push("--live-address".into());
    args.push(DISCARD_ADDRESS.into());
    args
}

/// Last `count` lines of a text.
pub fn tail_lines(text: &str, count: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(count);
    lines[start..].join("\n")
}

fn tail_chars(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        text.to_owned()
    } else {
        text.chars().skip(count - max_chars).collect()
    }
}

/// Reads the per-epoch stream the child's publisher writes. The child may be
/// mid-write, so only complete lines are considered.
pub fn read_live_metrics(path: &Path, candidate: &str) -> Vec<MetricPoint> {
    let Ok(bytes) = fs::read(path) else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&bytes);
    let complete = match text.rfind('\n') {
        Some(end) => &text[..=end],
        None => return Vec::new(),
    };
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(complete.as_bytes());
    reader
        .deserialize::<LiveMetric>()
        .filter_map(Result::ok)
        .filter(|metric| metric.candidate == candidate && metric.run == "search")
        .map(|metric| MetricPoint {
            epoch: metric.epoch,
            train_loss: metric.train_loss,
            validation_loss: metric.evaluation_loss,
            train_accuracy: metric.train_accuracy,
            validation_accuracy: metric.validation_accuracy,
        })
        .collect()
}

/// Finds the row of `candidate` in `candidate_summary.csv`, locating columns by
/// header name rather than position.
pub fn read_candidate_summary(run_dir: &Path, candidate: &str) -> Result<RunResult, String> {
    let path = run_dir.join("candidate_summary.csv");
    let mut reader = csv::Reader::from_path(&path)
        .map_err(|error| format!("could not read candidate_summary.csv: {error}"))?;
    let headers = reader
        .headers()
        .map_err(|error| format!("candidate_summary.csv has no header: {error}"))?
        .clone();
    let column = |name: &str| -> Result<usize, String> {
        headers
            .iter()
            .position(|header| header == name)
            .ok_or_else(|| format!("candidate_summary.csv has no column {name}"))
    };
    let name_column = column("name")?;
    let epochs_run = column("epochs_run")?;
    let best_epoch = column("best_epoch")?;
    let train_accuracy = column("train_accuracy_at_best")?;
    let validation_loss = column("validation_loss")?;
    let validation_accuracy = column("validation_accuracy")?;
    let parameter_count = column("parameter_count")?;
    let stopped_early = column("stopped_early")?;
    let seconds = column("seconds")?;
    for record in reader.records() {
        let record = record.map_err(|error| format!("bad candidate_summary.csv row: {error}"))?;
        if record.get(name_column) != Some(candidate) {
            continue;
        }
        let field = |index: usize| record.get(index).unwrap_or("").trim().to_owned();
        let number = |index: usize, label: &str| -> Result<f64, String> {
            field(index)
                .parse::<f64>()
                .map_err(|_| format!("candidate_summary.csv: bad {label}"))
        };
        let count = |index: usize, label: &str| -> Result<usize, String> {
            field(index)
                .parse::<usize>()
                .map_err(|_| format!("candidate_summary.csv: bad {label}"))
        };
        let train = number(train_accuracy, "train_accuracy_at_best")?;
        return Ok(RunResult {
            epochs_run: count(epochs_run, "epochs_run")?,
            best_epoch: count(best_epoch, "best_epoch")?,
            validation_accuracy: number(validation_accuracy, "validation_accuracy")?,
            validation_loss: number(validation_loss, "validation_loss")?,
            train_accuracy_at_best: train.is_finite().then_some(train),
            parameter_count: count(parameter_count, "parameter_count")?,
            stopped_early: field(stopped_early) == "true",
            seconds: number(seconds, "seconds")?,
        });
    }
    Err(format!("candidate_summary.csv has no row for {candidate}"))
}

impl RunManager {
    /// Starts `max_parallel` (at least one) worker threads.
    pub fn new(config: ManagerConfig) -> Arc<Self> {
        let workers = config.max_parallel.max(1);
        let session_secs = unix_ms() / 1000;
        let shared = Arc::new(Shared {
            config,
            session_secs,
            state: Mutex::new(State {
                runs: Vec::new(),
                queue: VecDeque::new(),
                running_by_group: HashMap::new(),
                next_seq: 1,
                next_group: 1,
            }),
            wakeup: Condvar::new(),
        });
        for index in 0..workers {
            let worker_shared = Arc::clone(&shared);
            let _ = thread::Builder::new()
                .name(format!("ui-worker-{index}"))
                .spawn(move || worker_loop(worker_shared));
        }
        Arc::new(Self { shared })
    }

    pub fn max_parallel(&self) -> usize {
        self.shared.config.max_parallel.max(1)
    }

    /// Server-generated id of a new sweep group: `g{session_secs}-{n}`.
    pub fn next_group_id(&self) -> String {
        let mut state = self.shared.lock();
        let id = format!(
            "g{}",
            format_run_id(self.shared.session_secs, state.next_group)
        );
        state.next_group += 1;
        id
    }

    fn preflight(&self, exercise: Exercise) -> Result<(), String> {
        let config = &self.shared.config;
        let required: Vec<(&Path, &str)> = match exercise {
            Exercise::Exercise2 => vec![(config.digits.as_path(), "--digits")],
            Exercise::Exercise3 => vec![
                (config.more_digits.as_path(), "--more-digits"),
                (config.digits.as_path(), "--digits"),
                (config.baseline_model.as_path(), "--baseline-model"),
            ],
        };
        for (path, flag) in required {
            if !path.is_file() {
                return Err(format!(
                    "{} not found: {} (set it with `tp3-rust ui {flag}`)",
                    exercise.as_str(),
                    path.display()
                ));
            }
        }
        Ok(())
    }

    /// Registers and enqueues one run per candidate. Errors are reported as
    /// messages for a 400 response.
    pub fn submit_batch(&self, batch: BatchSpec) -> Result<Vec<RunSummary>, String> {
        self.preflight(batch.exercise)?;
        let shared = &self.shared;
        let mut state = shared.lock();
        let active = state
            .runs
            .iter()
            .filter(|run| matches!(run.status, RunStatus::Queued | RunStatus::Running))
            .count();
        if active + batch.candidates.len() > MAX_QUEUED_RUNS {
            return Err(format!(
                "too many queued runs (limit {MAX_QUEUED_RUNS}); wait for some to finish"
            ));
        }
        fs::create_dir_all(&shared.config.output_root).map_err(|error| {
            format!(
                "could not create {}: {error}",
                shared.config.output_root.display()
            )
        })?;
        let mut created: Vec<(RunSummary, PathBuf)> = Vec::new();
        let mut seq = state.next_seq;
        let outcome: Result<(), String> = (|| {
            for candidate in &batch.candidates {
                let id = format_run_id(shared.session_secs, seq);
                seq += 1;
                let run_dir = shared.config.output_root.join(&id);
                fs::create_dir(&run_dir)
                    .map_err(|error| format!("could not create run directory: {error}"))?;
                let summary = RunSummary {
                    id,
                    group: batch.group.as_ref().map(|group| group.id.clone()),
                    exercise: batch.exercise,
                    name: candidate.name.clone(),
                    axis: candidate.axis.clone(),
                    status: RunStatus::Queued,
                    error: None,
                    last_epoch: None,
                    created_ms: unix_ms(),
                    started_ms: None,
                    finished_ms: None,
                    validation_ratio: batch.validation_ratio,
                    split_seed: batch.split_seed,
                    candidate: candidate.clone(),
                    result: None,
                };
                created.push((summary, run_dir.clone()));
                let toml = study_toml(batch.validation_ratio, batch.split_seed, candidate)?;
                fs::write(run_dir.join("config.toml"), toml)
                    .map_err(|error| format!("could not write config.toml: {error}"))?;
            }
            Ok(())
        })();
        if let Err(message) = outcome {
            for (_, dir) in &created {
                let _ = fs::remove_dir_all(dir);
            }
            return Err(message);
        }
        state.next_seq = seq;
        let mut summaries = Vec::new();
        for (summary, _) in created {
            state.queue.push_back(Job {
                run_id: summary.id.clone(),
                group: summary.group.clone(),
                parallel_cap: batch
                    .group
                    .as_ref()
                    .map_or(usize::MAX, |g| g.parallel.max(1)),
            });
            summaries.push(summary.clone());
            state.runs.push(summary);
        }
        drop(state);
        shared.wakeup.notify_all();
        Ok(summaries)
    }

    fn run_dir(&self, id: &str) -> PathBuf {
        self.shared.config.output_root.join(id)
    }

    fn fill_progress(&self, runs: &mut [RunSummary]) {
        for run in runs
            .iter_mut()
            .filter(|run| run.status == RunStatus::Running)
        {
            let points = read_live_metrics(
                &self.run_dir(&run.id).join("live_metrics.csv"),
                &run.candidate.name,
            );
            run.last_epoch = points.last().map(|point| point.epoch);
        }
    }

    /// Snapshot of every run, oldest first.
    pub fn list(&self) -> Vec<RunSummary> {
        let mut runs = self.shared.lock().runs.clone();
        self.fill_progress(&mut runs);
        runs
    }

    pub fn get(&self, id: &str) -> Option<RunSummary> {
        if !is_valid_id(id) {
            return None;
        }
        let mut runs: Vec<RunSummary> = self
            .shared
            .lock()
            .runs
            .iter()
            .filter(|run| run.id == id)
            .cloned()
            .collect();
        self.fill_progress(&mut runs);
        runs.pop()
    }

    pub fn metrics(&self, id: &str) -> Option<Vec<MetricPoint>> {
        let run = self.get(id)?;
        Some(read_live_metrics(
            &self.run_dir(&run.id).join("live_metrics.csv"),
            &run.candidate.name,
        ))
    }
}

fn worker_loop(shared: Arc<Shared>) {
    loop {
        let job = {
            let mut state = shared.lock();
            loop {
                let next = next_runnable(&state.queue, &state.running_by_group);
                if let Some(job) = next.and_then(|index| state.queue.remove(index)) {
                    if let Some(group) = &job.group {
                        *state.running_by_group.entry(group.clone()).or_insert(0) += 1;
                    }
                    if let Some(run) = state.runs.iter_mut().find(|run| run.id == job.run_id) {
                        run.status = RunStatus::Running;
                        run.started_ms = Some(unix_ms());
                    }
                    break job;
                }
                state = shared
                    .wakeup
                    .wait(state)
                    .unwrap_or_else(|error| error.into_inner());
            }
        };
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute_run(&shared, &job)))
                .unwrap_or_else(|_| Err("internal error while running the training".to_owned()));
        let mut state = shared.lock();
        if let Some(group) = &job.group {
            if let Some(count) = state.running_by_group.get_mut(group) {
                *count = count.saturating_sub(1);
            }
        }
        if let Some(run) = state.runs.iter_mut().find(|run| run.id == job.run_id) {
            run.finished_ms = Some(unix_ms());
            match outcome {
                Ok(result) => {
                    run.last_epoch = Some(result.epochs_run);
                    run.result = Some(result);
                    run.status = RunStatus::Done;
                }
                Err(message) => {
                    run.error = Some(message);
                    run.status = RunStatus::Failed;
                }
            }
        }
        drop(state);
        shared.wakeup.notify_all();
    }
}

/// Best finished run: highest validation accuracy, ties broken by lower
/// validation loss, fewer parameters, then smaller name (the same order as the
/// engine's `select_best_index`). Optionally scoped by exercise and group.
pub fn best_run<'a>(
    runs: &'a [RunSummary],
    exercise: Option<Exercise>,
    group: Option<&str>,
) -> Option<&'a RunSummary> {
    runs.iter()
        .filter(|run| run.status == RunStatus::Done && run.result.is_some())
        .filter(|run| exercise.is_none_or(|wanted| run.exercise == wanted))
        .filter(|run| group.is_none_or(|wanted| run.group.as_deref() == Some(wanted)))
        .max_by(|left, right| {
            let (Some(l), Some(r)) = (&left.result, &right.result) else {
                return std::cmp::Ordering::Equal;
            };
            l.validation_accuracy
                .total_cmp(&r.validation_accuracy)
                .then_with(|| r.validation_loss.total_cmp(&l.validation_loss))
                .then_with(|| r.parameter_count.cmp(&l.parameter_count))
                .then_with(|| right.name.cmp(&left.name))
        })
}

/// TOML download of a run: a comment header with its outcome followed by the
/// single-candidate study that reproduces it (`exercise2 train --config`).
pub fn export_toml(run: &RunSummary) -> Result<String, String> {
    let mut text = format!(
        "# tp3-rust ui export of run {} ({})\n",
        run.id,
        run.exercise.as_str()
    );
    if let Some(result) = &run.result {
        text.push_str(&format!(
            "# validation_accuracy = {}\n# validation_loss = {}\n# best_epoch = {}\n",
            result.validation_accuracy, result.validation_loss, result.best_epoch
        ));
    }
    text.push_str(&study_toml(
        run.validation_ratio,
        run.split_seed,
        &run.candidate,
    )?);
    Ok(text)
}

fn execute_run(shared: &Shared, job: &Job) -> Result<RunResult, String> {
    let id = job.run_id.as_str();
    let (exercise, name) = {
        let state = shared.lock();
        let run = state
            .runs
            .iter()
            .find(|run| run.id == id)
            .ok_or_else(|| "run vanished from the registry".to_owned())?;
        (run.exercise, run.candidate.name.clone())
    };
    let run_dir = shared.config.output_root.join(id);
    let stderr_path = run_dir.join("stderr.log");
    let stderr = File::create(&stderr_path)
        .map_err(|error| format!("could not create stderr.log: {error}"))?;
    let mut command = Command::new(&shared.config.exe);
    command
        .args(build_train_args(exercise, &shared.config, &run_dir))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr));
    if job.group.is_some() {
        // Concurrent sweep trainings share the cores instead of each spawning
        // one rayon thread per core.
        let cores = thread::available_parallelism().map_or(1, |cores| cores.get());
        command.env(
            "RAYON_NUM_THREADS",
            threads_per_run(cores, job.parallel_cap).to_string(),
        );
    }
    let status = command
        .status()
        .map_err(|error| format!("could not launch the training process: {error}"))?;
    if status.success() {
        read_candidate_summary(&run_dir, &name)
    } else {
        let text = fs::read(&stderr_path)
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_default();
        let tail = tail_chars(&tail_lines(&text, STDERR_TAIL_LINES), ERROR_MAX_CHARS);
        Err(if tail.trim().is_empty() {
            format!("training process exited with {status}")
        } else {
            tail
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digits::live_dashboard::LiveMetric;

    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("tp3-ui-{tag}-{}-{}", std::process::id(), unix_ms()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn manager_config() -> ManagerConfig {
        ManagerConfig {
            exe: PathBuf::from("/bin/tp3-rust"),
            output_root: PathBuf::from("out/ui"),
            digits: PathBuf::from("data/digits.csv"),
            more_digits: PathBuf::from("data/more digits.csv"),
            baseline_model: PathBuf::from("output/exercise2/selected_model.toml"),
            max_parallel: 2,
        }
    }

    fn metric(candidate: &str, run: &str, epoch: usize, validation: Option<f64>) -> LiveMetric {
        LiveMetric {
            session: "s".into(),
            exercise: "exercise2".into(),
            candidate: candidate.into(),
            run: run.into(),
            epoch,
            train_loss: 1.0 / epoch as f64,
            evaluation_loss: validation,
            train_accuracy: 0.5,
            validation_accuracy: validation.map(|_| 0.4),
            timestamp_ms: 1,
        }
    }

    #[test]
    fn run_ids() {
        assert_eq!(format_run_id(1759600000, 3), "1759600000-3");
        assert!(is_valid_id("1759600000-3"));
        for bad in ["", "-1", "1-", "1-2-3", "../x", "1;2", "abc", "1_2", "1-a"] {
            assert!(!is_valid_id(bad), "{bad:?}");
        }
    }

    #[test]
    fn exercise2_argv_is_exact() {
        let args = build_train_args(
            Exercise::Exercise2,
            &manager_config(),
            Path::new("out/ui/1-1"),
        );
        let expected: Vec<OsString> = [
            "exercise2",
            "train",
            "--data",
            "data/digits.csv",
            "--config",
            "out/ui/1-1/config.toml",
            "--output",
            "out/ui/1-1",
            "--live-publish-only",
            "--live-address",
            "127.0.0.1:9",
        ]
        .iter()
        .map(OsString::from)
        .collect();
        assert_eq!(args, expected);
    }

    #[test]
    fn exercise3_argv_keeps_paths_as_separate_elements() {
        let args = build_train_args(
            Exercise::Exercise3,
            &manager_config(),
            Path::new("out/ui/1-2"),
        );
        let text: Vec<String> = args
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(&text[..2], ["exercise3", "train"]);
        assert!(text.contains(&"--live-publish-only".to_owned()));
        let position = text.iter().position(|a| a == "--data").unwrap();
        // a path with a space stays one element
        assert_eq!(text[position + 1], "data/more digits.csv");
        let position = text.iter().position(|a| a == "--previous-data").unwrap();
        assert_eq!(text[position + 1], "data/digits.csv");
        let position = text.iter().position(|a| a == "--baseline-model").unwrap();
        assert_eq!(text[position + 1], "output/exercise2/selected_model.toml");
        assert!(!text
            .iter()
            .any(|a| a.starts_with("exercise3 ") || a.contains("--data ")));
    }

    #[test]
    fn tail_lines_keeps_the_end() {
        assert_eq!(tail_lines("a\nb\nc\nd", 2), "c\nd");
        assert_eq!(tail_lines("a\nb", 5), "a\nb");
        assert_eq!(tail_lines("", 3), "");
        assert_eq!(tail_chars("abcdef", 3), "def");
    }

    #[test]
    fn live_metrics_ignore_partial_lines_other_candidates_and_refit() {
        let dir = scratch("live");
        let path = dir.join("live_metrics.csv");
        let mut buffer = Vec::new();
        {
            let mut writer = csv::Writer::from_writer(&mut buffer);
            writer
                .serialize(metric("mine", "search", 1, Some(0.9)))
                .unwrap();
            writer
                .serialize(metric("other", "search", 1, Some(0.8)))
                .unwrap();
            writer.serialize(metric("mine", "refit", 2, None)).unwrap();
            writer.serialize(metric("mine", "search", 2, None)).unwrap();
            writer.flush().unwrap();
        }
        let mut text = String::from_utf8(buffer).unwrap();
        text.push_str("s,exercise2,mine,search,3,0.3,0.2,0.9");
        fs::write(&path, text).unwrap();
        let points = read_live_metrics(&path, "mine");
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].epoch, 1);
        assert_eq!(points[0].validation_loss, Some(0.9));
        assert_eq!(points[1].epoch, 2);
        assert_eq!(points[1].validation_loss, None);
        assert_eq!(points[1].validation_accuracy, None);
        assert!(read_live_metrics(&dir.join("missing.csv"), "mine").is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn candidate_summary_is_found_by_header_and_nan_becomes_none() {
        let dir = scratch("summary");
        let csv = "name,axis,topology,hidden_activation,optimizer,learning_rate,batch_size,initialization,weight_decay,dropout,lr_decay,augmentation,parameter_count,epochs_run,best_epoch,train_accuracy_at_best,validation_loss,validation_accuracy,stopped_early,seconds\n\
                   baseline,other,784-8-10,relu,adam,0.01,64,xavier,0,0,none,none,6370,10,8,0.95,0.30,0.91,true,12.5\n\
                   mine,other,784-8-10,relu,adam,0.01,64,xavier,0,0,none,none,6370,4,3,NaN,0.45,0.88,false,3.2\n";
        fs::write(dir.join("candidate_summary.csv"), csv).unwrap();
        let result = read_candidate_summary(&dir, "mine").unwrap();
        assert_eq!(result.epochs_run, 4);
        assert_eq!(result.best_epoch, 3);
        assert_eq!(result.train_accuracy_at_best, None);
        assert_eq!(result.validation_accuracy, 0.88);
        assert_eq!(result.validation_loss, 0.45);
        assert_eq!(result.parameter_count, 6370);
        assert!(!result.stopped_early);
        assert_eq!(result.seconds, 3.2);
        let baseline = read_candidate_summary(&dir, "baseline").unwrap();
        assert_eq!(baseline.train_accuracy_at_best, Some(0.95));
        assert!(baseline.stopped_early);
        let missing = read_candidate_summary(&dir, "ghost").unwrap_err();
        assert!(missing.contains("no row for ghost"));
        let _ = fs::remove_dir_all(dir);
    }

    fn job(id: &str, group: Option<&str>, cap: usize) -> Job {
        Job {
            run_id: id.into(),
            group: group.map(str::to_owned),
            parallel_cap: cap,
        }
    }

    #[test]
    fn group_ids_pass_the_gate() {
        assert!(is_valid_id("g1759600000-1"));
        for bad in ["g", "g-1", "gg1-1", "g1-", "G1-1", "g1-2-3"] {
            assert!(!is_valid_id(bad), "{bad:?}");
        }
    }

    #[test]
    fn next_runnable_is_fifo_and_skips_saturated_groups() {
        let mut running = HashMap::new();
        let mut queue: VecDeque<Job> = VecDeque::new();
        assert_eq!(next_runnable(&queue, &running), None);

        queue.push_back(job("1-1", Some("g1-1"), 2));
        queue.push_back(job("1-2", Some("g1-1"), 2));
        queue.push_back(job("1-3", None, usize::MAX));
        assert_eq!(next_runnable(&queue, &running), Some(0));

        // the group is full: the ungrouped job further back runs instead
        running.insert("g1-1".to_owned(), 2);
        assert_eq!(next_runnable(&queue, &running), Some(2));

        // everything queued belongs to the saturated group
        queue.pop_back();
        assert_eq!(next_runnable(&queue, &running), None);

        // below the cap again
        running.insert("g1-1".to_owned(), 1);
        assert_eq!(next_runnable(&queue, &running), Some(0));
    }

    #[test]
    fn threads_are_shared_between_concurrent_runs() {
        assert_eq!(threads_per_run(16, 2), 8);
        assert_eq!(threads_per_run(16, 4), 4);
        assert_eq!(threads_per_run(3, 4), 1);
        assert_eq!(threads_per_run(1, 8), 1);
        assert_eq!(threads_per_run(8, 0), 8);
        assert!(threads_per_run(0, 4) >= 1);
    }

    fn summary(
        id: &str,
        name: &str,
        exercise: Exercise,
        group: Option<&str>,
        outcome: Option<(f64, f64, usize)>,
    ) -> RunSummary {
        RunSummary {
            id: id.into(),
            group: group.map(str::to_owned),
            exercise,
            name: name.into(),
            axis: String::new(),
            status: if outcome.is_some() {
                RunStatus::Done
            } else {
                RunStatus::Running
            },
            error: None,
            last_epoch: None,
            created_ms: 0,
            started_ms: None,
            finished_ms: None,
            validation_ratio: 0.2,
            split_seed: 42,
            candidate: CandidateConfig {
                name: name.into(),
                ..CandidateConfig::default()
            },
            result: outcome.map(|(accuracy, loss, parameters)| RunResult {
                epochs_run: 5,
                best_epoch: 4,
                validation_accuracy: accuracy,
                validation_loss: loss,
                train_accuracy_at_best: Some(0.99),
                parameter_count: parameters,
                stopped_early: false,
                seconds: 1.0,
            }),
        }
    }

    #[test]
    fn best_run_orders_by_accuracy_then_loss_then_parameters_then_name() {
        let e2 = Exercise::Exercise2;
        let runs = vec![
            summary("1-1", "low", e2, None, Some((0.80, 0.5, 100))),
            summary("1-2", "high_loss", e2, None, Some((0.90, 0.6, 100))),
            summary("1-3", "low_loss", e2, None, Some((0.90, 0.4, 200))),
            summary("1-4", "small", e2, None, Some((0.90, 0.4, 100))),
            summary("1-5", "alpha", e2, None, Some((0.90, 0.4, 100))),
            summary("1-6", "running", e2, None, None),
        ];
        assert_eq!(best_run(&runs, None, None).unwrap().name, "alpha");
        assert!(best_run(&runs[5..], None, None).is_none());
        assert!(best_run(&[], None, None).is_none());
    }

    #[test]
    fn best_run_filters_by_group_and_exercise() {
        let runs = vec![
            summary(
                "1-1",
                "a",
                Exercise::Exercise2,
                Some("g1-1"),
                Some((0.7, 0.5, 1)),
            ),
            summary(
                "1-2",
                "b",
                Exercise::Exercise3,
                Some("g1-2"),
                Some((0.9, 0.5, 1)),
            ),
            summary("1-3", "c", Exercise::Exercise2, None, Some((0.8, 0.5, 1))),
        ];
        assert_eq!(best_run(&runs, None, None).unwrap().name, "b");
        assert_eq!(
            best_run(&runs, Some(Exercise::Exercise2), None)
                .unwrap()
                .name,
            "c"
        );
        assert_eq!(best_run(&runs, None, Some("g1-1")).unwrap().name, "a");
        assert!(best_run(&runs, Some(Exercise::Exercise3), Some("g1-1")).is_none());
        assert!(best_run(&runs, None, Some("g9-9")).is_none());
    }

    #[test]
    fn export_toml_carries_a_comment_header_and_a_valid_study() {
        use crate::digits::config::DigitStudyConfig;
        let run = summary(
            "1-7",
            "exported",
            Exercise::Exercise2,
            None,
            Some((0.9, 0.4, 100)),
        );
        let text = export_toml(&run).unwrap();
        let header: Vec<&str> = text
            .lines()
            .take_while(|line| line.starts_with('#'))
            .collect();
        assert!(header[0].contains("1-7"));
        assert!(header
            .iter()
            .any(|l| l.contains("validation_accuracy = 0.9")));
        assert!(header.iter().any(|l| l.contains("validation_loss = 0.4")));
        assert!(header.iter().any(|l| l.contains("best_epoch = 4")));
        let study: DigitStudyConfig = toml::from_str(&text).unwrap();
        study.validate().unwrap();
        assert_eq!(study.candidates.len(), 1);
        assert_eq!(study.candidates[0].name, "exported");
        let pending = summary("1-8", "pending", Exercise::Exercise2, None, None);
        assert!(export_toml(&pending)
            .unwrap()
            .starts_with("# tp3-rust ui export"));
    }
}
