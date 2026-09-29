mod multi_layer_perceptron;
mod single_layer;
mod step_perceptron;

use thiserror::Error;

use crate::{
    matrix::DenseMatrix,
    model::{ModelError, SingleLayerPerceptron},
};

pub use multi_layer_perceptron::*;
pub use single_layer::*;
pub use step_perceptron::*;

#[derive(Clone, Copy, Debug)]
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub shuffle: bool,
    pub seed: u64,
    pub early_stopping: Option<EarlyStopping>,
}

impl TrainingConfig {
    pub fn fixed(learning_rate: f64, max_epochs: usize, seed: u64) -> Self {
        Self {
            learning_rate,
            max_epochs,
            shuffle: true,
            seed,
            early_stopping: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub validation_loss: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct TrainingReport {
    pub history: Vec<EpochMetrics>,
    pub best_epoch: usize,
    pub best_monitored_loss: f64,
    pub stopped_early: bool,
}

#[derive(Debug, Error)]
pub enum TrainingError {
    #[error("training configuration values must be finite and positive")]
    InvalidConfig,
    #[error("training indices cannot be empty")]
    EmptyTrainingSet,
    #[error("target count {targets} does not match matrix rows {rows}")]
    TargetLength { targets: usize, rows: usize },
    #[error("sample index {index} is out of bounds")]
    InvalidIndex { index: usize },
    #[error("step perceptron targets must be either -1 or 1")]
    InvalidStepTarget,
    #[error("MLP scalar training requires exactly one output")]
    NonScalarOutput,
    #[error("training produced a non-finite loss")]
    NonFiniteLoss,
    #[error(transparent)]
    Model(#[from] ModelError),
}

fn validate_common(
    features: &DenseMatrix,
    target_count: usize,
    train_indices: &[usize],
    config: TrainingConfig,
) -> Result<(), TrainingError> {
    if !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.max_epochs == 0
        || config.early_stopping.is_some_and(|stopping| {
            stopping.patience == 0 || !stopping.min_delta.is_finite() || stopping.min_delta < 0.0
        })
    {
        return Err(TrainingError::InvalidConfig);
    }
    if target_count != features.rows() {
        return Err(TrainingError::TargetLength {
            targets: target_count,
            rows: features.rows(),
        });
    }
    if train_indices.is_empty() {
        return Err(TrainingError::EmptyTrainingSet);
    }
    validate_indices(features, train_indices)
}

fn validate_indices(features: &DenseMatrix, indices: &[usize]) -> Result<(), TrainingError> {
    if let Some(&index) = indices.iter().find(|&&index| index >= features.rows()) {
        return Err(TrainingError::InvalidIndex { index });
    }
    Ok(())
}

#[cfg(feature = "training-logs")]
fn print_epoch_loss(model: &str, learning_rate: f64, metrics: &EpochMetrics) {
    if let Some(validation_loss) = metrics.validation_loss {
        eprintln!(
            "model={model} learning_rate={learning_rate} epoch={} train_loss={:.10} validation_loss={:.10}",
            metrics.epoch, metrics.train_loss, validation_loss
        );
    } else {
        eprintln!(
            "model={model} learning_rate={learning_rate} epoch={} train_loss={:.10}",
            metrics.epoch, metrics.train_loss
        );
    }
}

pub fn predict_single_indices(
    model: &SingleLayerPerceptron,
    features: &DenseMatrix,
    indices: &[usize],
) -> Result<Vec<f64>, TrainingError> {
    use rayon::prelude::*;

    validate_indices(features, indices)?;
    indices
        .par_iter()
        .map(|&index| Ok(model.predict(features.row_unchecked(index))?))
        .collect()
}
