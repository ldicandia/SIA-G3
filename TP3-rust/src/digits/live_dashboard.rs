mod monitor;
mod publisher;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use monitor::*;
pub use publisher::*;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LiveMetric {
    pub session: String,
    pub exercise: String,
    pub candidate: String,
    pub run: String,
    pub epoch: usize,
    pub train_loss: f64,
    pub evaluation_loss: Option<f64>,
    pub train_accuracy: f64,
    pub validation_accuracy: Option<f64>,
    pub timestamp_ms: u64,
}

#[derive(Debug, Error)]
pub enum LiveError {
    #[error("could not initialize live metrics: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_metric_json_round_trip_preserves_optional_evaluation_values() {
        let metric = LiveMetric {
            session: "s".into(),
            exercise: "exercise2".into(),
            candidate: "candidate".into(),
            run: "search".into(),
            epoch: 3,
            train_loss: 0.2,
            evaluation_loss: Some(0.3),
            train_accuracy: 0.9,
            validation_accuracy: Some(0.8),
            timestamp_ms: 1,
        };
        let decoded: LiveMetric =
            serde_json::from_slice(&serde_json::to_vec(&metric).unwrap()).unwrap();
        assert_eq!(decoded.epoch, 3);
        assert_eq!(decoded.evaluation_loss, Some(0.3));
    }
}
