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

/// One optimizer step with batch-mean `gradients` and this epoch's learning
/// rate. Weight decay is decoupled (AdamW style): weights shrink by
/// `learning_rate * weight_decay` independently of the adaptive update, and
/// biases are not decayed.
pub(super) fn apply_gradients(
    model: &mut MultilayerPerceptron,
    gradients: &[LayerValues],
    state: &mut OptimizerState,
    candidate: &CandidateConfig,
    learning_rate: f64,
) {
    state.step += 1;
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
            learning_rate,
            candidate.weight_decay,
        );
        update_values(
            &mut layer.biases,
            &gradient.biases,
            &mut first.biases,
            &mut second.biases,
            candidate,
            state.step,
            learning_rate,
            0.0,
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
    learning_rate: f64,
    weight_decay: f64,
) {
    let decay = 1.0 - learning_rate * weight_decay;
    let (bias1, bias2) = (
        1.0 - candidate.beta1.powi(step as i32),
        1.0 - candidate.beta2.powi(step as i32),
    );
    for index in 0..parameters.len() {
        let gradient = gradients[index];
        if weight_decay > 0.0 {
            parameters[index] *= decay;
        }
        match candidate.optimizer {
            OptimizerKind::Sgd => parameters[index] -= learning_rate * gradient,
            OptimizerKind::Momentum => {
                first[index] = candidate.momentum * first[index] + gradient;
                parameters[index] -= learning_rate * first[index];
            }
            OptimizerKind::Adam => {
                first[index] = candidate.beta1 * first[index] + (1.0 - candidate.beta1) * gradient;
                second[index] =
                    candidate.beta2 * second[index] + (1.0 - candidate.beta2) * gradient * gradient;
                parameters[index] -= learning_rate * (first[index] / bias1)
                    / ((second[index] / bias2).sqrt() + candidate.epsilon);
            }
        }
    }
}
