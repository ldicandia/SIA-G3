mod augment;
mod backprop;
mod optimizer;

use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use thiserror::Error;

use crate::{
    matrix::DenseMatrix,
    model::{Activation, ModelError, MultilayerPerceptron},
};

use super::{config::CandidateConfig, data::DIGIT_CLASSES, live_dashboard::LiveMetricsPublisher};

use backprop::*;
use optimizer::*;

/// Rows per parallel shard when scoring a whole dataset.
const EVALUATION_SHARD: usize = 64;
/// Smallest training shard; smaller shards spend more time on bookkeeping.
const MIN_TRAINING_SHARD: usize = 4;
/// Parameters summed per parallel task when reducing shard gradients.
const REDUCTION_CHUNK: usize = 8192;

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
    pub seconds: f64,
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
        TrainingRun {
            train_indices,
            validation_indices: Some(validation_indices),
            candidate,
            epochs: candidate.max_epochs,
            epoch_offset: 0,
            early_stopping: true,
            run: "search",
            publisher,
        },
    )
}

/// Trains from the model's current weights for a fixed number of epochs,
/// without validation. `epoch_offset` continues the learning-rate schedule and
/// random streams of a previous run, so a saved model can resume training.
#[allow(clippy::too_many_arguments)]
pub fn refit_digit_model(
    model: &mut MultilayerPerceptron,
    features: &DenseMatrix,
    labels: &[usize],
    indices: &[usize],
    candidate: &CandidateConfig,
    epochs: usize,
    epoch_offset: usize,
    run: &str,
    publisher: Option<&LiveMetricsPublisher>,
) -> Result<DigitTrainingReport, DigitTrainingError> {
    train(
        model,
        features,
        labels,
        TrainingRun {
            train_indices: indices,
            validation_indices: None,
            candidate,
            epochs,
            epoch_offset,
            early_stopping: false,
            run,
            publisher,
        },
    )
}

struct TrainingRun<'a> {
    train_indices: &'a [usize],
    validation_indices: Option<&'a [usize]>,
    candidate: &'a CandidateConfig,
    epochs: usize,
    epoch_offset: usize,
    early_stopping: bool,
    run: &'a str,
    publisher: Option<&'a LiveMetricsPublisher>,
}

fn train(
    model: &mut MultilayerPerceptron,
    features: &DenseMatrix,
    labels: &[usize],
    settings: TrainingRun<'_>,
) -> Result<DigitTrainingReport, DigitTrainingError> {
    let started = std::time::Instant::now();
    let candidate = settings.candidate;
    validate_inputs(model, features, labels, settings.train_indices)?;
    if let Some(indices) = settings.validation_indices {
        validate_indices(features, indices)?;
    }

    // A batch is split into shards that run forward/backward in parallel,
    // each with its own buffers; their gradients are then summed.
    let shard_size = candidate
        .batch_size
        .div_ceil(rayon::current_num_threads().max(1))
        .max(MIN_TRAINING_SHARD);
    let mut shards = (0..candidate.batch_size.div_ceil(shard_size))
        .map(|_| BatchWorkspace::new(model, shard_size, true))
        .collect::<Vec<_>>();
    let mut gradients = model
        .layers
        .iter()
        .map(|layer| LayerValues {
            weights: vec![0.0; layer.weights.len()],
            biases: vec![0.0; layer.biases.len()],
        })
        .collect::<Vec<_>>();
    let mut optimizer = OptimizerState::new(model);
    let mut order = settings.train_indices.to_vec();
    let mut rng =
        ChaCha8Rng::seed_from_u64(candidate.seed.wrapping_add(settings.epoch_offset as u64));
    let mut history = Vec::with_capacity(settings.epochs);
    let mut best_model = model.clone();
    let mut best_epoch = 0;
    let mut best_loss = f64::INFINITY;
    let mut best_accuracy = 0.0;
    let mut stale_epochs = 0;
    let mut stopped_early = false;

    for local_epoch in 1..=settings.epochs {
        let epoch = settings.epoch_offset + local_epoch;
        let learning_rate = candidate.learning_rate_at(epoch);
        order.shuffle(&mut rng);
        let augmentation = candidate.augments().then_some((candidate, epoch));
        let dropout = (candidate.dropout > 0.0).then_some(Dropout {
            probability: candidate.dropout,
            seed: candidate.seed,
            epoch,
        });
        for batch in order.chunks(candidate.batch_size) {
            let pieces = batch.chunks(shard_size).collect::<Vec<_>>();
            let active = &mut shards[..pieces.len()];
            let frozen: &MultilayerPerceptron = model;
            active
                .par_iter_mut()
                .zip(pieces.par_iter())
                .for_each(|(workspace, rows)| {
                    load_inputs(workspace, features, rows, augmentation);
                    forward_batch(frozen, workspace, rows, dropout.as_ref());
                    backward_batch(
                        frozen,
                        workspace,
                        labels,
                        rows,
                        batch.len(),
                        dropout.is_some(),
                    );
                });
            sum_shard_gradients(active, &mut gradients);
            apply_gradients(model, &gradients, &mut optimizer, candidate, learning_rate);
        }

        let train_evaluation =
            evaluate_digit_model(model, features, labels, settings.train_indices)?;
        let validation_evaluation = settings
            .validation_indices
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
        if let Some(publisher) = settings.publisher {
            publisher.publish(&candidate.name, settings.run, history.last().unwrap());
        }

        if monitored_loss + candidate.min_delta < best_loss {
            best_loss = monitored_loss;
            best_accuracy = monitored_accuracy;
            best_epoch = epoch;
            if settings.early_stopping {
                best_model = model.clone();
            }
            stale_epochs = 0;
        } else {
            stale_epochs += 1;
        }
        if settings.early_stopping && stale_epochs >= candidate.patience {
            stopped_early = true;
            break;
        }
    }

    if settings.early_stopping {
        *model = best_model;
    }
    Ok(DigitTrainingReport {
        history,
        best_epoch,
        best_validation_loss: best_loss,
        best_validation_accuracy: best_accuracy,
        stopped_early,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// Mean cross-entropy, accuracy and predictions, computed in batched forward
/// passes without dropout or augmentation.
pub fn evaluate_digit_model(
    model: &MultilayerPerceptron,
    features: &DenseMatrix,
    labels: &[usize],
    indices: &[usize],
) -> Result<DigitEvaluation, DigitTrainingError> {
    validate_inputs(model, features, labels, indices)?;
    let probabilities = predict_digit_probabilities(model, features, indices)?;
    let mut loss = 0.0;
    let mut correct = 0;
    let mut predictions = Vec::with_capacity(indices.len());
    for (&index, row) in indices
        .iter()
        .zip(probabilities.as_chunks::<DIGIT_CLASSES>().0)
    {
        let prediction = argmax(row);
        loss -= row[labels[index]].max(f64::MIN_POSITIVE).ln();
        correct += usize::from(prediction == labels[index]);
        predictions.push(prediction);
    }
    if !loss.is_finite() {
        return Err(DigitTrainingError::NonFinite);
    }
    Ok(DigitEvaluation {
        loss: loss / indices.len() as f64,
        accuracy: correct as f64 / indices.len() as f64,
        predictions,
    })
}

/// Softmax probabilities for every row, used by ensembles.
pub fn predict_digit_probabilities(
    model: &MultilayerPerceptron,
    features: &DenseMatrix,
    indices: &[usize],
) -> Result<Vec<f64>, DigitTrainingError> {
    validate_indices(features, indices)?;
    if indices.is_empty() {
        return Err(DigitTrainingError::EmptyTrainingSet);
    }
    let shards = indices
        .par_chunks(EVALUATION_SHARD)
        .map_init(
            || BatchWorkspace::new(model, EVALUATION_SHARD, false),
            |workspace, rows| {
                load_inputs(workspace, features, rows, None);
                forward_batch(model, workspace, rows, None);
                workspace.probabilities(rows.len(), DIGIT_CLASSES).to_vec()
            },
        )
        .collect::<Vec<_>>();
    Ok(shards.concat())
}

/// `target = sum of the shard gradients`, reduced in parallel chunks.
fn sum_shard_gradients(shards: &[BatchWorkspace], target: &mut [LayerValues]) {
    for (layer, values) in target.iter_mut().enumerate() {
        values
            .weights
            .par_chunks_mut(REDUCTION_CHUNK)
            .enumerate()
            .for_each(|(chunk, output)| {
                let start = chunk * REDUCTION_CHUNK;
                let end = start + output.len();
                output.copy_from_slice(&shards[0].gradients[layer].weights[start..end]);
                for shard in &shards[1..] {
                    for (sum, &value) in output
                        .iter_mut()
                        .zip(&shard.gradients[layer].weights[start..end])
                    {
                        *sum += value;
                    }
                }
            });
        values
            .biases
            .copy_from_slice(&shards[0].gradients[layer].biases);
        for shard in &shards[1..] {
            for (sum, &value) in values.biases.iter_mut().zip(&shard.gradients[layer].biases) {
                *sum += value;
            }
        }
    }
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
    use crate::{digits::config::OptimizerKind, model::softmax_in_place};

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
            ..CandidateConfig::default()
        }
    }

    fn tiny_model(config: &CandidateConfig) -> MultilayerPerceptron {
        MultilayerPerceptron::new(
            &config.topology,
            &[Activation::Tanh, Activation::Linear],
            config.seed,
        )
        .unwrap()
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
            let mut model = tiny_model(&config);
            let before = evaluate_digit_model(&model, &features, &labels, &[0])
                .unwrap()
                .loss;
            refit_digit_model(
                &mut model,
                &features,
                &labels,
                &[0],
                &config,
                1,
                0,
                "test",
                None,
            )
            .unwrap();
            let after = evaluate_digit_model(&model, &features, &labels, &[0])
                .unwrap()
                .loss;
            assert!(after < before, "{optimizer:?}: {before} -> {after}");
        }
    }

    /// The batched backward pass must match a finite-difference gradient.
    #[test]
    fn batch_gradient_matches_finite_differences() {
        let features =
            DenseMatrix::from_rows(vec![vec![0.3, -0.7], vec![-0.2, 0.5], vec![0.9, 0.1]]).unwrap();
        let labels = [1, 4, 7];
        let batch = [0, 1, 2];
        let config = candidate(OptimizerKind::Sgd);
        let model = tiny_model(&config);
        // Two shards of a three-row batch must sum to the full-batch gradient.
        let mut first = BatchWorkspace::new(&model, 2, true);
        let mut second = BatchWorkspace::new(&model, 2, true);
        for (workspace, rows) in [(&mut first, &batch[..2]), (&mut second, &batch[2..])] {
            load_inputs(workspace, &features, rows, None);
            forward_batch(&model, workspace, rows, None);
            backward_batch(&model, workspace, &labels, rows, batch.len(), false);
        }
        let mut gradients = model
            .layers
            .iter()
            .map(|layer| LayerValues {
                weights: vec![0.0; layer.weights.len()],
                biases: vec![0.0; layer.biases.len()],
            })
            .collect::<Vec<_>>();
        sum_shard_gradients(&[first, second], &mut gradients);

        let mean_loss = |model: &MultilayerPerceptron| {
            batch
                .iter()
                .map(|&row| {
                    let mut output = model.predict(features.row(row).unwrap()).unwrap();
                    softmax_in_place(&mut output);
                    -output[labels[row]].ln()
                })
                .sum::<f64>()
                / batch.len() as f64
        };
        let epsilon = 1e-6;
        for (layer, gradient) in gradients.iter().enumerate() {
            for weight in 0..gradient.weights.len() {
                let mut plus = model.clone();
                plus.layers[layer].weights[weight] += epsilon;
                let mut minus = model.clone();
                minus.layers[layer].weights[weight] -= epsilon;
                let numerical = (mean_loss(&plus) - mean_loss(&minus)) / (2.0 * epsilon);
                let analytical = gradient.weights[weight];
                assert!(
                    (numerical - analytical).abs() < 1e-6,
                    "layer {layer} weight {weight}: {numerical} vs {analytical}"
                );
            }
        }
    }
}
