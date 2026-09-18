use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Linear,
    Sigmoid,
    Tanh,
}

impl Activation {
    pub fn apply(self, value: f64) -> f64 {
        match self {
            Self::Linear => value,
            Self::Sigmoid if value >= 0.0 => 1.0 / (1.0 + (-value).exp()),
            Self::Sigmoid => {
                let exp = value.exp();
                exp / (1.0 + exp)
            }
            Self::Tanh => value.tanh(),
        }
    }

    pub fn derivative_from_output(self, output: f64) -> f64 {
        match self {
            Self::Linear => 1.0,
            Self::Sigmoid => output * (1.0 - output),
            Self::Tanh => 1.0 - output.powi(2),
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModelError {
    #[error("model expects {expected} inputs, got {actual}")]
    InputDimension { expected: usize, actual: usize },
    #[error("a model must have at least one input")]
    EmptyInput,
    #[error("an MLP needs at least an input and output size")]
    InvalidTopology,
    #[error("the activation count must match the number of layers")]
    InvalidActivationCount,
}

#[derive(Clone, Debug)]
pub struct StepPerceptron {
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: f64,
}

impl StepPerceptron {
    pub fn new(input_size: usize) -> Result<Self, ModelError> {
        if input_size == 0 {
            return Err(ModelError::EmptyInput);
        }
        Ok(Self {
            weights: vec![0.0; input_size],
            bias: 0.0,
        })
    }

    pub fn predict(&self, input: &[f64]) -> Result<i8, ModelError> {
        Ok(if self.score(input)? >= 0.0 { 1 } else { -1 })
    }

    pub fn score(&self, input: &[f64]) -> Result<f64, ModelError> {
        check_input(input, self.weights.len())?;
        Ok(dot(&self.weights, input) + self.bias)
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }
}

#[derive(Clone, Debug)]
pub struct SingleLayerPerceptron {
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: f64,
    activation: Activation,
}

impl SingleLayerPerceptron {
    pub fn new(input_size: usize, activation: Activation, seed: u64) -> Result<Self, ModelError> {
        if input_size == 0 {
            return Err(ModelError::EmptyInput);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let limit = (6.0 / (input_size + 1) as f64).sqrt();
        let weights = (0..input_size)
            .map(|_| rng.gen_range(-limit..=limit))
            .collect();
        Ok(Self {
            weights,
            bias: 0.0,
            activation,
        })
    }

    pub fn predict(&self, input: &[f64]) -> Result<f64, ModelError> {
        check_input(input, self.weights.len())?;
        Ok(self.activation.apply(dot(&self.weights, input) + self.bias))
    }

    pub fn input_size(&self) -> usize {
        self.weights.len()
    }

    pub fn activation(&self) -> Activation {
        self.activation
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DenseLayer {
    pub(crate) input_size: usize,
    pub(crate) output_size: usize,
    pub(crate) weights: Vec<f64>,
    pub(crate) biases: Vec<f64>,
    pub(crate) activation: Activation,
}

#[derive(Clone, Debug)]
pub struct MultilayerPerceptron {
    pub(crate) layers: Vec<DenseLayer>,
}

impl MultilayerPerceptron {
    pub fn new(
        topology: &[usize],
        activations: &[Activation],
        seed: u64,
    ) -> Result<Self, ModelError> {
        if topology.len() < 2 || topology.contains(&0) {
            return Err(ModelError::InvalidTopology);
        }
        if activations.len() != topology.len() - 1 {
            return Err(ModelError::InvalidActivationCount);
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let layers = topology
            .windows(2)
            .zip(activations.iter().copied())
            .map(|(sizes, activation)| {
                let input_size = sizes[0];
                let output_size = sizes[1];
                let limit = (6.0 / (input_size + output_size) as f64).sqrt();
                DenseLayer {
                    input_size,
                    output_size,
                    weights: (0..input_size * output_size)
                        .map(|_| rng.gen_range(-limit..=limit))
                        .collect(),
                    biases: vec![0.0; output_size],
                    activation,
                }
            })
            .collect();
        Ok(Self { layers })
    }

    pub fn predict(&self, input: &[f64]) -> Result<Vec<f64>, ModelError> {
        let expected = self.layers[0].input_size;
        check_input(input, expected)?;
        let mut current = input.to_vec();
        for layer in &self.layers {
            let mut next = vec![0.0; layer.output_size];
            for (output, value) in next.iter_mut().enumerate() {
                let start = output * layer.input_size;
                *value = layer.activation.apply(
                    dot(&layer.weights[start..start + layer.input_size], &current)
                        + layer.biases[output],
                );
            }
            current = next;
        }
        Ok(current)
    }

    pub fn topology(&self) -> Vec<usize> {
        let mut topology = vec![self.layers[0].input_size];
        topology.extend(self.layers.iter().map(|layer| layer.output_size));
        topology
    }
}

pub(crate) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn check_input(input: &[f64], expected: usize) -> Result<(), ModelError> {
    if input.len() != expected {
        return Err(ModelError::InputDimension {
            expected,
            actual: input.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn activation_derivatives_match_finite_differences() {
        let epsilon = 1e-6;
        for activation in [Activation::Linear, Activation::Sigmoid, Activation::Tanh] {
            let value = 0.37;
            let output = activation.apply(value);
            let numerical = (activation.apply(value + epsilon) - activation.apply(value - epsilon))
                / (2.0 * epsilon);
            assert_abs_diff_eq!(
                activation.derivative_from_output(output),
                numerical,
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn stable_sigmoid_handles_large_inputs() {
        assert_abs_diff_eq!(Activation::Sigmoid.apply(1000.0), 1.0);
        assert_abs_diff_eq!(Activation::Sigmoid.apply(-1000.0), 0.0);
    }

    #[test]
    fn rejects_wrong_prediction_dimensions() {
        let model = SingleLayerPerceptron::new(2, Activation::Linear, 1).unwrap();
        assert!(matches!(
            model.predict(&[1.0]),
            Err(ModelError::InputDimension { .. })
        ));
    }
}
