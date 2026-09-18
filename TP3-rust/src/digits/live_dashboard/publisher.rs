use std::{
    fs::File,
    io::BufWriter,
    net::{SocketAddr, UdpSocket},
    path::Path,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, SyncSender, TrySendError},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{SystemTime, UNIX_EPOCH},
};

use super::{LiveError, LiveMetric};
use crate::digits::training::DigitEpochMetrics;

const QUEUE_CAPACITY: usize = 256;

enum PublisherCommand {
    Metric(LiveMetric),
    Finish,
}

pub struct LiveMetricsPublisher {
    exercise: String,
    session: String,
    sender: Option<SyncSender<PublisherCommand>>,
    worker: Option<JoinHandle<()>>,
    dropped: Arc<AtomicUsize>,
}

impl LiveMetricsPublisher {
    pub fn disabled(exercise: &str) -> Self {
        Self {
            exercise: exercise.to_owned(),
            session: String::new(),
            sender: None,
            worker: None,
            dropped: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn new(exercise: &str, output: &Path, target: SocketAddr) -> Result<Self, LiveError> {
        let file = File::create(output.join("live_metrics.csv"))?;
        let socket = UdpSocket::bind("127.0.0.1:0")?;
        socket.set_nonblocking(true)?;
        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        let dropped = Arc::new(AtomicUsize::new(0));
        let session = format!("{}-{}", std::process::id(), unix_time_ms());
        let worker = thread::spawn(move || {
            let mut writer = csv::Writer::from_writer(BufWriter::new(file));
            while let Ok(command) = receiver.recv() {
                match command {
                    PublisherCommand::Metric(metric) => {
                        let _ = writer.serialize(&metric);
                        let _ = writer.flush();
                        if let Ok(payload) = serde_json::to_vec(&metric) {
                            let _ = socket.send_to(&payload, target);
                        }
                    }
                    PublisherCommand::Finish => break,
                }
            }
            let _ = writer.flush();
        });
        Ok(Self {
            exercise: exercise.to_owned(),
            session,
            sender: Some(sender),
            worker: Some(worker),
            dropped,
        })
    }

    pub fn publish(&self, candidate: &str, run: &str, metrics: &DigitEpochMetrics) {
        let Some(sender) = &self.sender else {
            return;
        };
        let metric = LiveMetric {
            session: self.session.clone(),
            exercise: self.exercise.clone(),
            candidate: candidate.to_owned(),
            run: run.to_owned(),
            epoch: metrics.epoch,
            train_loss: metrics.train_loss,
            evaluation_loss: metrics.validation_loss,
            train_accuracy: metrics.train_accuracy,
            validation_accuracy: metrics.validation_accuracy,
            timestamp_ms: unix_time_ms(),
        };
        if let Err(error) = sender.try_send(PublisherCommand::Metric(metric)) {
            if matches!(error, TrySendError::Full(_)) {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn dropped_events(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }
}

pub fn spawn_monitor_process(
    udp_address: SocketAddr,
    http_address: SocketAddr,
) -> Result<u32, LiveError> {
    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg("monitor")
        .arg("--udp-address")
        .arg(udp_address.to_string())
        .arg("--http-address")
        .arg(http_address.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let child = command.spawn()?;
    Ok(child.id())
}

impl Drop for LiveMetricsPublisher {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(PublisherCommand::Finish);
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_publisher_drops_metrics_without_queueing_work() {
        let publisher = LiveMetricsPublisher::disabled("exercise2");
        let metrics = DigitEpochMetrics {
            epoch: 1,
            train_loss: 0.4,
            validation_loss: Some(0.5),
            train_accuracy: 0.8,
            validation_accuracy: Some(0.7),
        };

        publisher.publish("candidate", "search", &metrics);

        assert_eq!(publisher.dropped_events(), 0);
        assert!(publisher.sender.is_none());
        assert!(publisher.worker.is_none());
    }
}
