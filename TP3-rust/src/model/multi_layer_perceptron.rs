use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::{check_input, dot, Activation, ModelError};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DenseLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
    pub activation: Activation,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MultilayerPerceptron {
    pub layers: Vec<DenseLayer>,
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

    pub fn predict_probabilities(&self, input: &[f64]) -> Result<Vec<f64>, ModelError> {
        let mut probabilities = self.predict(input)?;
        softmax_in_place(&mut probabilities);
        Ok(probabilities)
    }

    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.weights.len() + layer.biases.len())
            .sum()
    }
}

pub fn softmax_in_place(values: &mut [f64]) {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        total += *value;
    }
    for value in values {
        *value /= total;
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn softmax_is_stable_and_normalized() {
        let mut values = vec![1000.0, 1001.0, 999.0];
        softmax_in_place(&mut values);
        assert!(values.iter().all(|value| value.is_finite()));
        assert_abs_diff_eq!(values.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
        assert_eq!(
            values
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0,
            1
        );
    }
}
