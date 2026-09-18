mod backprop;
mod optimizer;

use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

use crate::{
    loss::categorical_cross_entropy,
    matrix::DenseMatrix,
    model::{Activation, ModelError, MultilayerPerceptron},
};

use super::{config::CandidateConfig, data::DIGIT_CLASSES, live_dashboard::LiveMetricsPublisher};

use backprop::*;
use optimizer::*;

#[derive(Clone, Debug)]
pub struct DigitEpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub validation_loss: Option<f64>,
    pub train_accuracy: f64,
    pub validation_accuracy: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct DigitTrainingReport {
    pub history: Vec<DigitEpochMetrics>,
    pub best_epoch: usize,
    pub best_validation_loss: f64,
    pub best_validation_accuracy: f64,
    pub stopped_early: bool,
}

#[derive(Clone, Debug)]
pub struct DigitEvaluation {
    pub loss: f64,
    pub accuracy: f64,
    pub predictions: Vec<usize>,
}

#[derive(Debug, Error)]
pub enum DigitTrainingError {
    #[error("training indices cannot be empty")]
    EmptyTrainingSet,
    #[error("labels and matrix rows must match")]
    LabelLength,
    #[error("sample index {0} is out of bounds")]
    InvalidIndex(usize),
    #[error("model topology is incompatible with the digit dataset")]
    InvalidTopology,
    #[error("training produced a non-finite value")]
    NonFinite,
    #[error(transparent)]
    Model(#[from] ModelError),
}

pub fn train_digit_candidate(
    model: &mut MultilayerPerceptron,
    features: &DenseMatrix,
    labels: &[usize],
    train_indices: &[usize],
    validation_indices: &[usize],
    candidate: &CandidateConfig,
    publisher: Option<&LiveMetricsPublisher>,
) -> Result<DigitTrainingReport, DigitTrainingError> {
    train(
        model,
        features,
        labels,
        train_indices,
        Some(validation_indices),
        candidate,
        candidate.max_epochs,
        true,
        "search",
        publisher,
    )
}

pub fn refit_digit_model(
    model: &mut MultilayerPerceptron,
    features: &DenseMatrix,
    labels: &[usize],
    indices: &[usize],
    candidate: &CandidateConfig,
    epochs: usize,
    publisher: Option<&LiveMetricsPublisher>,
) -> Result<DigitTrainingReport, DigitTrainingError> {
    train(
        model, features, labels, indices, None, candidate, epochs, false, "refit", publisher,
    )
}

#[allow(clippy::too_many_arguments)]
fn train(
    model: &mut MultilayerPerceptron,
    features: &DenseMatrix,
    labels: &[usize],
    train_indices: &[usize],
    validation_indices: Option<&[usize]>,
    candidate: &CandidateConfig,
    epochs: usize,
    early_stopping: bool,
    run: &str,
    publisher: Option<&LiveMetricsPublisher>,
) -> Result<DigitTrainingReport, DigitTrainingError> {
    validate_inputs(model, features, labels, train_indices)?;
    if let Some(indices) = validation_indices {
        validate_indices(features, indices)?;
    }

    let topology = model.topology();
    let mut gradients = model
        .layers
        .iter()
        .map(|layer| LayerValues {
            weights: vec![0.0; layer.weights.len()],
            biases: vec![0.0; layer.biases.len()],
        })
        .collect::<Vec<_>>();
    let mut optimizer = OptimizerState::new(model);
    let mut order = train_indices.to_vec();
    let mut rng = ChaCha8Rng::seed_from_u64(candidate.seed);
    let mut history = Vec::with_capacity(epochs);
    let mut best_model = model.clone();
    let mut best_epoch = 0;
    let mut best_loss = f64::INFINITY;
    let mut best_accuracy = 0.0;
    let mut stale_epochs = 0;
    let mut stopped_early = false;

    for epoch in 1..=epochs {
        order.shuffle(&mut rng);
        for batch in order.chunks(candidate.batch_size) {
            fill_batch_gradients(model, features, labels, batch, &topology, &mut gradients);
            apply_gradients(model, &gradients, &mut optimizer, candidate, batch.len());
        }

        let train_evaluation = evaluate_digit_model(model, features, labels, train_indices)?;
        let validation_evaluation = validation_indices
            .map(|indices| evaluate_digit_model(model, features, labels, indices))
            .transpose()?;
        let monitored_loss = validation_evaluation
            .as_ref()
            .map_or(train_evaluation.loss, |evaluation| evaluation.loss);
        let monitored_accuracy = validation_evaluation
            .as_ref()
            .map_or(train_evaluation.accuracy, |evaluation| evaluation.accuracy);
        if !monitored_loss.is_finite() {
            return Err(DigitTrainingError::NonFinite);
        }
        history.push(DigitEpochMetrics {
            epoch,
            train_loss: train_evaluation.loss,
            validation_loss: validation_evaluation
                .as_ref()
                .map(|evaluation| evaluation.loss),
            train_accuracy: train_evaluation.accuracy,
            validation_accuracy: validation_evaluation
                .as_ref()
                .map(|evaluation| evaluation.accuracy),
        });
        print_epoch(&candidate.name, history.last().unwrap());
        if let Some(publisher) = publisher {
            publisher.publish(&candidate.name, run, history.last().unwrap());
        }

        if monitored_loss + candidate.min_delta < best_loss {
            best_loss = monitored_loss;
            best_accuracy = monitored_accuracy;
            best_epoch = epoch;
            best_model = model.clone();
            stale_epochs = 0;
        } else {
            stale_epochs += 1;
        }
        if early_stopping && stale_epochs >= candidate.patience {
            stopped_early = true;
            break;
        }
    }

    if early_stopping {
        *model = best_model;
    }
    Ok(DigitTrainingReport {
        history,
        best_epoch,
        best_validation_loss: best_loss,
        best_validation_accuracy: best_accuracy,
        stopped_early,
    })
}

fn fill_batch_gradients(
    model: &MultilayerPerceptron,
    features: &DenseMatrix,
    labels: &[usize],
    batch: &[usize],
    topology: &[usize],
    gradients: &mut [LayerValues],
) {
    use rayon::prelude::*;

    let zeroed_gradients = || {
        model
            .layers
            .iter()
            .map(|layer| LayerValues {
                weights: vec![0.0; layer.weights.len()],
                biases: vec![0.0; layer.biases.len()],
            })
            .collect::<Vec<_>>()
    };
    let reduced = batch
        .par_iter()
        .map_init(
            || {
                let activations = topology.iter().map(|&size| vec![0.0; size]).collect();
                let deltas = topology[1..].iter().map(|&size| vec![0.0; size]).collect();
                (activations, deltas)
            },
            |(activations, deltas): &mut (Vec<Vec<f64>>, Vec<Vec<f64>>), &index| {
                activations[0].copy_from_slice(features.row_unchecked(index));
                forward_softmax(model, activations);
                backward(model, activations, deltas, labels[index]);
                let mut sample_gradients = zeroed_gradients();
                accumulate_gradients(activations, deltas, &mut sample_gradients);
                sample_gradients
            },
        )
        .reduce(zeroed_gradients, |mut left, right| {
            add_layer_values(&mut left, &right);
            left
        });
    zero_gradients(gradients);
    add_layer_values(gradients, &reduced);
}

pub fn evaluate_digit_model(
    model: &MultilayerPerceptron,
    features: &DenseMatrix,
    labels: &[usize],
    indices: &[usize],
) -> Result<DigitEvaluation, DigitTrainingError> {
    use rayon::prelude::*;

    validate_inputs(model, features, labels, indices)?;
    let topology = model.topology();
    let per_sample: Vec<(f64, usize)> = indices
        .par_iter()
        .map_init(
            || {
                topology
                    .iter()
                    .map(|&size| vec![0.0; size])
                    .collect::<Vec<_>>()
            },
            |activations, &index| {
                activations[0].copy_from_slice(features.row_unchecked(index));
                forward_softmax(model, activations);
                let probabilities = activations.last().unwrap();
                let sample_loss = categorical_cross_entropy(probabilities, labels[index])
                    .map_err(|_| DigitTrainingError::NonFinite)?;
                Ok((sample_loss, argmax(probabilities)))
            },
        )
        .collect::<Result<_, DigitTrainingError>>()?;

    let mut loss = 0.0;
    let mut correct = 0;
    let mut predictions = Vec::with_capacity(indices.len());
    for (&index, &(sample_loss, prediction)) in indices.iter().zip(&per_sample) {
        loss += sample_loss;
        correct += usize::from(prediction == labels[index]);
        predictions.push(prediction);
    }
    Ok(DigitEvaluation {
        loss: loss / indices.len() as f64,
        accuracy: correct as f64 / indices.len() as f64,
        predictions,
    })
}

fn validate_inputs(
    model: &MultilayerPerceptron,
    features: &DenseMatrix,
    labels: &[usize],
    indices: &[usize],
) -> Result<(), DigitTrainingError> {
    if indices.is_empty() {
        return Err(DigitTrainingError::EmptyTrainingSet);
    }
    if labels.len() != features.rows() {
        return Err(DigitTrainingError::LabelLength);
    }
    validate_indices(features, indices)?;
    let topology = model.topology();
    if topology.first() != Some(&features.cols())
        || topology.last() != Some(&DIGIT_CLASSES)
        || model.layers.last().map(|layer| layer.activation) != Some(Activation::Linear)
    {
        return Err(DigitTrainingError::InvalidTopology);
    }
    Ok(())
}

fn validate_indices(features: &DenseMatrix, indices: &[usize]) -> Result<(), DigitTrainingError> {
    if let Some(&index) = indices.iter().find(|&&index| index >= features.rows()) {
        Err(DigitTrainingError::InvalidIndex(index))
    } else {
        Ok(())
    }
}

#[cfg(feature = "training-logs")]
fn print_epoch(candidate: &str, metrics: &DigitEpochMetrics) {
    if let (Some(validation_loss), Some(validation_accuracy)) =
        (metrics.validation_loss, metrics.validation_accuracy)
    {
        eprintln!(
            "candidate={candidate} epoch={} train_loss={:.8} validation_loss={:.8} train_accuracy={:.6} validation_accuracy={:.6}",
            metrics.epoch,
            metrics.train_loss,
            validation_loss,
            metrics.train_accuracy,
            validation_accuracy
        );
    } else {
        eprintln!(
            "candidate={candidate} epoch={} train_loss={:.8} train_accuracy={:.6}",
            metrics.epoch, metrics.train_loss, metrics.train_accuracy
        );
    }
}

#[cfg(not(feature = "training-logs"))]
fn print_epoch(_candidate: &str, _metrics: &DigitEpochMetrics) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digits::config::OptimizerKind;

    fn candidate(optimizer: OptimizerKind) -> CandidateConfig {
        CandidateConfig {
            name: "tiny".into(),
            topology: vec![2, 3, 10],
            hidden_activation: Activation::Tanh,
            optimizer,
            learning_rate: 0.05,
            batch_size: 1,
            max_epochs: 1,
            patience: 1,
            min_delta: 0.0,
            seed: 2,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    #[test]
    fn one_update_reduces_cross_entropy_for_each_optimizer() {
        let features = DenseMatrix::from_rows(vec![vec![0.2, -0.4]]).unwrap();
        let labels = [3];
        for optimizer in [
            OptimizerKind::Sgd,
            OptimizerKind::Momentum,
            OptimizerKind::Adam,
        ] {
            let config = candidate(optimizer);
            let mut model = MultilayerPerceptron::new(
                &config.topology,
                &[Activation::Tanh, Activation::Linear],
                config.seed,
            )
            .unwrap();
            let before = evaluate_digit_model(&model, &features, &labels, &[0])
                .unwrap()
                .loss;
            refit_digit_model(&mut model, &features, &labels, &[0], &config, 1, None).unwrap();
            let after = evaluate_digit_model(&model, &features, &labels, &[0])
                .unwrap()
                .loss;
            assert!(after < before, "{optimizer:?}: {before} -> {after}");
        }
    }
}
