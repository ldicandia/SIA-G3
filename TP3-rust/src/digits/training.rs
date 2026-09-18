use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

use crate::{
    loss::categorical_cross_entropy,
    matrix::DenseMatrix,
    model::{softmax_in_place, Activation, ModelError, MultilayerPerceptron},
};

use super::{
    config::{CandidateConfig, OptimizerKind},
    data::DIGIT_CLASSES,
    live::LiveMetricsPublisher,
};

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

#[derive(Clone)]
struct LayerValues {
    weights: Vec<f64>,
    biases: Vec<f64>,
}

struct OptimizerState {
    first: Vec<LayerValues>,
    second: Vec<LayerValues>,
    step: usize,
}

impl OptimizerState {
    fn new(model: &MultilayerPerceptron) -> Self {
        let zeros = || {
            model
                .layers
                .iter()
                .map(|layer| LayerValues {
                    weights: vec![0.0; layer.weights.len()],
                    biases: vec![0.0; layer.biases.len()],
                })
                .collect::<Vec<_>>()
        };
        Self {
            first: zeros(),
            second: zeros(),
            step: 0,
        }
    }
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

fn forward_softmax(model: &MultilayerPerceptron, activations: &mut [Vec<f64>]) {
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let (before, after) = activations.split_at_mut(layer_index + 1);
        let input = &before[layer_index];
        let output = &mut after[0];
        for (output_index, output_value) in output.iter_mut().enumerate() {
            let start = output_index * layer.input_size;
            let value = layer.weights[start..start + layer.input_size]
                .iter()
                .zip(input)
                .map(|(weight, input)| weight * input)
                .sum::<f64>()
                + layer.biases[output_index];
            *output_value = layer.activation.apply(value);
        }
    }
    softmax_in_place(activations.last_mut().unwrap());
}

fn backward(
    model: &MultilayerPerceptron,
    activations: &[Vec<f64>],
    deltas: &mut [Vec<f64>],
    target: usize,
) {
    let last = model.layers.len() - 1;
    deltas[last].copy_from_slice(activations.last().unwrap());
    deltas[last][target] -= 1.0;

    for layer_index in (0..last).rev() {
        let next_layer = &model.layers[layer_index + 1];
        let (current, next) = deltas.split_at_mut(layer_index + 1);
        for neuron in 0..model.layers[layer_index].output_size {
            let propagated = (0..next_layer.output_size)
                .map(|output| {
                    next_layer.weights[output * next_layer.input_size + neuron] * next[0][output]
                })
                .sum::<f64>();
            current[layer_index][neuron] = propagated
                * model.layers[layer_index]
                    .activation
                    .derivative_from_output(activations[layer_index + 1][neuron]);
        }
    }
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

fn zero_gradients(gradients: &mut [LayerValues]) {
    for layer in gradients {
        layer.weights.fill(0.0);
        layer.biases.fill(0.0);
    }
}

fn add_layer_values(left: &mut [LayerValues], right: &[LayerValues]) {
    for (left_layer, right_layer) in left.iter_mut().zip(right) {
        for (l, r) in left_layer.weights.iter_mut().zip(&right_layer.weights) {
            *l += r;
        }
        for (l, r) in left_layer.biases.iter_mut().zip(&right_layer.biases) {
            *l += r;
        }
    }
}

fn accumulate_gradients(
    activations: &[Vec<f64>],
    deltas: &[Vec<f64>],
    gradients: &mut [LayerValues],
) {
    for (layer_index, gradient) in gradients.iter_mut().enumerate() {
        let input = &activations[layer_index];
        for (output, &delta) in deltas[layer_index].iter().enumerate() {
            let start = output * input.len();
            for (weight_gradient, &value) in gradient.weights[start..start + input.len()]
                .iter_mut()
                .zip(input)
            {
                *weight_gradient += delta * value;
            }
            gradient.biases[output] += delta;
        }
    }
}

fn apply_gradients(
    model: &mut MultilayerPerceptron,
    gradients: &[LayerValues],
    state: &mut OptimizerState,
    candidate: &CandidateConfig,
    batch_size: usize,
) {
    state.step += 1;
    let scale = 1.0 / batch_size as f64;
    for (((layer, gradient), first), second) in model
        .layers
        .iter_mut()
        .zip(gradients)
        .zip(&mut state.first)
        .zip(&mut state.second)
    {
        update_values(
            &mut layer.weights,
            &gradient.weights,
            &mut first.weights,
            &mut second.weights,
            candidate,
            state.step,
            scale,
        );
        update_values(
            &mut layer.biases,
            &gradient.biases,
            &mut first.biases,
            &mut second.biases,
            candidate,
            state.step,
            scale,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn update_values(
    parameters: &mut [f64],
    gradients: &[f64],
    first: &mut [f64],
    second: &mut [f64],
    candidate: &CandidateConfig,
    step: usize,
    scale: f64,
) {
    for index in 0..parameters.len() {
        let gradient = gradients[index] * scale;
        match candidate.optimizer {
            OptimizerKind::Sgd => parameters[index] -= candidate.learning_rate * gradient,
            OptimizerKind::Momentum => {
                first[index] = candidate.momentum * first[index] + gradient;
                parameters[index] -= candidate.learning_rate * first[index];
            }
            OptimizerKind::Adam => {
                first[index] = candidate.beta1 * first[index] + (1.0 - candidate.beta1) * gradient;
                second[index] =
                    candidate.beta2 * second[index] + (1.0 - candidate.beta2) * gradient * gradient;
                let first_corrected = first[index] / (1.0 - candidate.beta1.powi(step as i32));
                let second_corrected = second[index] / (1.0 - candidate.beta2.powi(step as i32));
                parameters[index] -= candidate.learning_rate * first_corrected
                    / (second_corrected.sqrt() + candidate.epsilon);
            }
        }
    }
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
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
