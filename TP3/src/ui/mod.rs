//! Local web UI for the digit exercises: a loopback-only HTTP server that
//! launches `tp3-rust exercise{2|3} train` subprocesses and serves their
//! progress. Standard library networking only, no new crates.

mod http;
mod runs;
mod spec;

use std::{
    io::Write,
    net::{SocketAddr, TcpListener, TcpStream},
    path::PathBuf,
    sync::Arc,
    thread,
    time::Duration,
};

use anyhow::{bail, Context, Result};

use crate::digits::config::CandidateConfig;
use http::{check_guards, read_request, HttpError, Request, Response};
use runs::{
    best_run, export_toml, is_valid_id, BatchSpec, GroupSpec, ManagerConfig, RunManager, RunSummary,
};
use spec::{
    default_split_seed, default_validation_ratio, plan_request, Exercise, RunRequest,
    DEFAULT_SWEEP_PARALLEL, SWEEP_FIELDS,
};

const IO_TIMEOUT: Duration = Duration::from_secs(5);
const INDEX_HTML: &str = include_str!("index.html");
const APP_JS: &str = include_str!("app.js");
/// The page needs no inline script, so `script-src` falls back to 'self'.
const PAGE_CSP: &str = "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'";

#[derive(Clone, Debug)]
pub struct UiConfig {
    pub addr: SocketAddr,
    pub exe: PathBuf,
    pub output_root: PathBuf,
    pub digits: PathBuf,
    pub more_digits: PathBuf,
    pub baseline_model: PathBuf,
    pub max_parallel: usize,
}

pub struct UiServer {
    local_addr: SocketAddr,
}

impl UiServer {
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }
}

struct AppState {
    manager: Arc<RunManager>,
    port: u16,
}

/// Binds the listener and starts serving in background threads, then returns.
pub fn start(config: UiConfig) -> Result<UiServer> {
    if !config.addr.ip().is_loopback() {
        bail!(
            "refusing to bind {}: the UI launches processes, so it only binds to loopback",
            config.addr
        );
    }
    let manager = RunManager::new(ManagerConfig {
        exe: config.exe,
        output_root: config.output_root,
        digits: config.digits,
        more_digits: config.more_digits,
        baseline_model: config.baseline_model,
        max_parallel: config.max_parallel.max(1),
    });
    let listener = TcpListener::bind(config.addr)
        .with_context(|| format!("could not bind {}", config.addr))?;
    let local_addr = listener.local_addr()?;
    let state = Arc::new(AppState {
        manager,
        port: local_addr.port(),
    });
    thread::Builder::new()
        .name("ui-accept".into())
        .spawn(move || {
            for stream in listener.incoming().flatten() {
                let state = Arc::clone(&state);
                let _ = thread::Builder::new()
                    .name("ui-connection".into())
                    .spawn(move || handle_connection(stream, &state));
            }
        })
        .context("could not start the accept thread")?;
    Ok(UiServer { local_addr })
}

/// Serves the UI until the process is stopped.
pub fn run_ui(config: UiConfig) -> Result<()> {
    let server = start(config)?;
    eprintln!("tp3-rust ui: http://{}", server.local_addr());
    loop {
        thread::park();
    }
}

fn handle_connection(mut stream: TcpStream, state: &AppState) {
    let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
    let response = match read_request(&mut stream) {
        Ok(request) => match check_guards(&request, state.port) {
            Ok(()) => route(&request, state),
            Err(response) => response,
        },
        Err(HttpError::Io) => return,
        Err(error) => Response::error(error.status(), error.message()),
    };
    let _ = stream.write_all(&response.to_bytes());
    let _ = stream.flush();
    let _ = stream.shutdown(std::net::Shutdown::Both);
}

fn not_found() -> Response {
    Response::error(404, "not found")
}

fn method_not_allowed() -> Response {
    Response::error(405, "method not allowed")
}

fn route(request: &Request, state: &AppState) -> Response {
    let segments: Vec<&str> = request.path.trim_matches('/').split('/').collect();
    let get = request.method == "GET";
    match segments.as_slice() {
        [""] | ["index.html"] => {
            if get {
                Response::text(200, "text/html; charset=utf-8", INDEX_HTML.to_owned())
                    .with_header("Content-Security-Policy", PAGE_CSP)
            } else {
                method_not_allowed()
            }
        }
        ["app.js"] => {
            if get {
                Response::text(200, "text/javascript; charset=utf-8", APP_JS.to_owned())
            } else {
                method_not_allowed()
            }
        }
        ["api", "defaults"] => {
            if get {
                defaults(state)
            } else {
                method_not_allowed()
            }
        }
        ["api", "runs"] => match request.method.as_str() {
            "GET" => Response::json(200, &state.manager.list()),
            "POST" => create_runs(request, state),
            _ => method_not_allowed(),
        },
        ["api", "runs", "best", "config.toml"] => {
            if get {
                best_export(request, state)
            } else {
                method_not_allowed()
            }
        }
        ["api", "runs", id, "config.toml"] => {
            if !get {
                method_not_allowed()
            } else if !is_valid_id(id) {
                not_found()
            } else {
                match state.manager.get(id) {
                    Some(run) => toml_download(&run),
                    None => not_found(),
                }
            }
        }
        ["api", "runs", id] => {
            if !get {
                method_not_allowed()
            } else if !is_valid_id(id) {
                not_found()
            } else {
                match state.manager.get(id) {
                    Some(run) => Response::json(200, &run),
                    None => not_found(),
                }
            }
        }
        ["api", "runs", id, "metrics"] => {
            if !get {
                method_not_allowed()
            } else if !is_valid_id(id) {
                not_found()
            } else {
                match state.manager.metrics(id) {
                    Some(points) => Response::json(200, &points),
                    None => not_found(),
                }
            }
        }
        _ => not_found(),
    }
}

fn defaults(state: &AppState) -> Response {
    Response::json(
        200,
        &serde_json::json!({
            "candidate": CandidateConfig::default(),
            "validation_ratio": default_validation_ratio(),
            "split_seed": default_split_seed(),
            "exercises": [Exercise::Exercise2.as_str(), Exercise::Exercise3.as_str()],
            "max_parallel": state.manager.max_parallel(),
            "sweep_fields": SWEEP_FIELDS,
            "default_parallel": DEFAULT_SWEEP_PARALLEL.min(state.manager.max_parallel()),
        }),
    )
}

/// Run names are restricted to letters, digits, '_' and '-', so the header
/// cannot be broken by them.
fn toml_download(run: &RunSummary) -> Response {
    match export_toml(run) {
        Ok(text) => Response::text(200, "application/toml; charset=utf-8", text).with_header(
            "Content-Disposition",
            &format!("attachment; filename=\"{}.toml\"", run.name),
        ),
        Err(message) => Response::error(500, &message),
    }
}

fn best_export(request: &Request, state: &AppState) -> Response {
    let group = match request.query_param("group") {
        Some(group) if is_valid_id(group) => Some(group),
        Some(_) => return Response::error(400, "invalid group id"),
        None => None,
    };
    let exercise = match request.query_param("exercise") {
        Some(text) => match Exercise::parse(text) {
            Some(exercise) => Some(exercise),
            None => return Response::error(400, "exercise must be exercise2 or exercise3"),
        },
        None => None,
    };
    let runs = state.manager.list();
    match best_run(&runs, exercise, group) {
        Some(run) => toml_download(run),
        None => Response::error(404, "no finished run matches"),
    }
}

fn create_runs(request: &Request, state: &AppState) -> Response {
    let run_request: RunRequest = match serde_json::from_slice(&request.body) {
        Ok(parsed) => parsed,
        Err(error) => {
            let message: String = error.to_string().chars().take(200).collect();
            return Response::error(400, &format!("invalid request body: {message}"));
        }
    };
    let planned = match plan_request(&run_request, state.manager.max_parallel()) {
        Ok(planned) => planned,
        Err(message) => return Response::error(400, &message),
    };
    let group = planned.parallel.map(|parallel| GroupSpec {
        id: state.manager.next_group_id(),
        parallel,
    });
    let group_id = group.as_ref().map(|group| group.id.clone());
    let batch = BatchSpec {
        exercise: run_request.exercise,
        validation_ratio: run_request.validation_ratio,
        split_seed: run_request.split_seed,
        candidates: planned.candidates,
        group,
    };
    match state.manager.submit_batch(batch) {
        Ok(runs) => Response::json(202, &serde_json::json!({ "group": group_id, "runs": runs })),
        Err(message) => Response::error(400, &message),
    }
}
