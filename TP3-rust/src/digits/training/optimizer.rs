use crate::{
    digits::config::{CandidateConfig, OptimizerKind},
    model::MultilayerPerceptron,
};

#[derive(Clone)]
pub(super) struct LayerValues {
    pub(super) weights: Vec<f64>,
    pub(super) biases: Vec<f64>,
}

pub(super) struct OptimizerState {
    first: Vec<LayerValues>,
    second: Vec<LayerValues>,
    step: usize,
}

impl OptimizerState {
    pub(super) fn new(model: &MultilayerPerceptron) -> Self {
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

pub(super) fn zero_gradients(gradients: &mut [LayerValues]) {
    for layer in gradients {
        layer.weights.fill(0.0);
        layer.biases.fill(0.0);
    }
}

pub(super) fn add_layer_values(left: &mut [LayerValues], right: &[LayerValues]) {
    for (left_layer, right_layer) in left.iter_mut().zip(right) {
        for (l, r) in left_layer.weights.iter_mut().zip(&right_layer.weights) {
            *l += r;
        }
        for (l, r) in left_layer.biases.iter_mut().zip(&right_layer.biases) {
            *l += r;
        }
    }
}

pub(super) fn apply_gradients(
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
