//! Mini-batch forward and backward passes expressed as matrix products.
//!
//! For a batch of `B` samples, layer `l` holds its activations as a `B x n_l`
//! row-major matrix. With softmax + cross-entropy the output delta is
//! `(P - Y) / B`, and each earlier delta is `(D W) ⊙ f'(A) ⊙ mask`.
//!
//! These functions process one shard (a slice of the batch) sequentially;
//! `training.rs` runs the shards of a batch in parallel and sums their
//! gradients, which equals the gradient of the whole batch.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    matrix::{affine_nt, matmul_nn, matmul_tn, DenseMatrix},
    model::{softmax_in_place, MultilayerPerceptron},
};

use super::{super::config::CandidateConfig, augment::augment_into, optimizer::LayerValues};

/// Buffers for one shard, sized for its largest row count, plus the shard's
/// gradient accumulator. Allocated once per training run.
pub(super) struct BatchWorkspace {
    activations: Vec<Vec<f64>>,
    masks: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
    pub(super) gradients: Vec<LayerValues>,
}

/// Inverted dropout settings for one training step.
pub(super) struct Dropout {
    pub(super) probability: f64,
    pub(super) seed: u64,
    pub(super) epoch: usize,
}

impl BatchWorkspace {
    /// `with_gradients` is false for evaluation-only workspaces.
    pub(super) fn new(model: &MultilayerPerceptron, rows: usize, with_gradients: bool) -> Self {
        let topology = model.topology();
        Self {
            activations: topology
                .iter()
                .map(|&size| vec![0.0; rows * size])
                .collect(),
            masks: topology[1..topology.len() - 1]
                .iter()
                .map(|&size| vec![0.0; rows * size])
                .collect(),
            deltas: topology[1..]
                .iter()
                .map(|&size| vec![0.0; rows * size])
                .collect(),
            gradients: if with_gradients {
                model
                    .layers
                    .iter()
                    .map(|layer| LayerValues {
                        weights: vec![0.0; layer.weights.len()],
                        biases: vec![0.0; layer.biases.len()],
                    })
                    .collect()
            } else {
                Vec::new()
            },
        }
    }

    /// Softmax outputs of the first `rows` samples after `forward_batch`.
    pub(super) fn probabilities(&self, rows: usize, classes: usize) -> &[f64] {
        &self.activations.last().unwrap()[..rows * classes]
    }
}

/// Deterministic per-sample random stream, independent of how the batch is
/// split into shards or scheduled on threads.
fn sample_rng(seed: u64, epoch: usize, index: usize, salt: u64) -> ChaCha8Rng {
    let mixed = seed
        ^ (epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (index as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ salt.wrapping_mul(0x1656_67B1_9E37_79F9);
    ChaCha8Rng::seed_from_u64(mixed)
}

/// Copies the shard rows into the input activations, augmenting them when
/// training with a candidate that asks for it.
pub(super) fn load_inputs(
    workspace: &mut BatchWorkspace,
    features: &DenseMatrix,
    rows: &[usize],
    augmentation: Option<(&CandidateConfig, usize)>,
) {
    let cols = features.cols();
    let input = &mut workspace.activations[0][..rows.len() * cols];
    for (row, &index) in input.chunks_exact_mut(cols).zip(rows) {
        let source = features.row_unchecked(index);
        match augmentation {
            Some((candidate, epoch)) => {
                let mut rng = sample_rng(candidate.seed, epoch, index, 0);
                augment_into(
                    source,
                    row,
                    &mut rng,
                    candidate.augment_shift,
                    candidate.augment_rotation,
                );
            }
            None => row.copy_from_slice(source),
        }
    }
}

/// Forward pass for the shard `rows`; the output layer ends in softmax.
pub(super) fn forward_batch(
    model: &MultilayerPerceptron,
    workspace: &mut BatchWorkspace,
    rows: &[usize],
    dropout: Option<&Dropout>,
) {
    let count = rows.len();
    let last = model.layers.len() - 1;
    for (index, layer) in model.layers.iter().enumerate() {
        let (before, after) = workspace.activations.split_at_mut(index + 1);
        let input = &before[index][..count * layer.input_size];
        let output = &mut after[0][..count * layer.output_size];
        affine_nt(
            input,
            layer.input_size,
            &layer.weights,
            &layer.biases,
            output,
        );
        if index == last {
            output
                .chunks_exact_mut(layer.output_size)
                .for_each(softmax_in_place);
            continue;
        }
        let activation = layer.activation;
        match dropout {
            None => output
                .iter_mut()
                .for_each(|value| *value = activation.apply(*value)),
            Some(settings) => {
                let keep = 1.0 - settings.probability;
                let mask = &mut workspace.masks[index][..count * layer.output_size];
                for ((values, factors), &sample) in output
                    .chunks_exact_mut(layer.output_size)
                    .zip(mask.chunks_exact_mut(layer.output_size))
                    .zip(rows)
                {
                    let mut rng =
                        sample_rng(settings.seed, settings.epoch, sample, 1 + index as u64);
                    for (value, factor) in values.iter_mut().zip(factors.iter_mut()) {
                        *factor = if rng.gen::<f64>() < keep {
                            1.0 / keep
                        } else {
                            0.0
                        };
                        *value = activation.apply(*value) * *factor;
                    }
                }
            }
        }
    }
}

/// Backward pass for the shard `rows`, writing into `workspace.gradients`.
/// Deltas are divided by `batch_rows` (the whole batch, not the shard), so
/// summing the shards' gradients yields the batch-mean gradient.
pub(super) fn backward_batch(
    model: &MultilayerPerceptron,
    workspace: &mut BatchWorkspace,
    labels: &[usize],
    rows: &[usize],
    batch_rows: usize,
    dropout: bool,
) {
    let count = rows.len();
    let last = model.layers.len() - 1;
    let classes = model.layers[last].output_size;
    let inverse = 1.0 / batch_rows as f64;
    for ((delta, probabilities), &index) in workspace.deltas[last][..count * classes]
        .chunks_exact_mut(classes)
        .zip(workspace.activations[last + 1][..count * classes].chunks_exact(classes))
        .zip(rows)
    {
        for (value, &probability) in delta.iter_mut().zip(probabilities) {
            *value = probability * inverse;
        }
        delta[labels[index]] -= inverse;
    }

    for index in (0..=last).rev() {
        let layer = &model.layers[index];
        let (inputs, outputs) = (layer.input_size, layer.output_size);
        let gradient = &mut workspace.gradients[index];
        matmul_tn(
            &workspace.deltas[index][..count * outputs],
            outputs,
            &workspace.activations[index][..count * inputs],
            inputs,
            &mut gradient.weights,
            &mut gradient.biases,
        );
        if index == 0 {
            break;
        }
        let (lower, upper) = workspace.deltas.split_at_mut(index);
        let previous = &mut lower[index - 1][..count * inputs];
        matmul_nn(
            &upper[0][..count * outputs],
            outputs,
            &layer.weights,
            previous,
        );
        let activation = model.layers[index - 1].activation;
        let values = &workspace.activations[index][..count * inputs];
        if dropout {
            let mask = &workspace.masks[index - 1][..count * inputs];
            for ((delta, &value), &factor) in previous.iter_mut().zip(values).zip(mask) {
                *delta = if factor == 0.0 {
                    0.0
                } else {
                    *delta * factor * activation.derivative_from_output(value / factor)
                };
            }
        } else {
            for (delta, &value) in previous.iter_mut().zip(values) {
                *delta *= activation.derivative_from_output(value);
            }
        }
    }
}

/// Gradient of one logit per row with respect to the network input, run
/// after `forward_batch(model, workspace, rows, None)` on the same shard.
///
/// The output delta is a one-hot vector at each row's target class: the
/// gradient of the target LOGIT `z_c`, not of softmax cross-entropy, so the
/// softmax stored in the last activation is ignored. It is then propagated
/// with the same `D · W` products and `f'(A)` factors as `backward_batch`,
/// one step further than training does: at layer 0 the product `D · W₀` is
/// `∂z_c / ∂x`. No dropout and no parameter gradients. Writes
/// `count x input_size` values into `input_gradients`.
pub(super) fn backward_input_batch(
    model: &MultilayerPerceptron,
    workspace: &mut BatchWorkspace,
    target_classes: &[usize],
    count: usize,
    input_gradients: &mut [f64],
) {
    let last = model.layers.len() - 1;
    let classes = model.layers[last].output_size;
    let output_delta = &mut workspace.deltas[last][..count * classes];
    output_delta.fill(0.0);
    for (delta, &target) in output_delta.chunks_exact_mut(classes).zip(target_classes) {
        delta[target] = 1.0;
    }

    for index in (0..=last).rev() {
        let layer = &model.layers[index];
        let (inputs, outputs) = (layer.input_size, layer.output_size);
        if index == 0 {
            matmul_nn(
                &workspace.deltas[0][..count * outputs],
                outputs,
                &layer.weights,
                &mut input_gradients[..count * inputs],
            );
            break;
        }
        let (lower, upper) = workspace.deltas.split_at_mut(index);
        let previous = &mut lower[index - 1][..count * inputs];
        matmul_nn(
            &upper[0][..count * outputs],
            outputs,
            &layer.weights,
            previous,
        );
        let activation = model.layers[index - 1].activation;
        let values = &workspace.activations[index][..count * inputs];
        for (delta, &value) in previous.iter_mut().zip(values) {
            *delta *= activation.derivative_from_output(value);
        }
    }
}

pub(super) fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}
