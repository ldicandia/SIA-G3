use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::{check_input, dot, Activation, ModelError};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_prediction_dimensions() {
        let model = SingleLayerPerceptron::new(2, Activation::Linear, 1).unwrap();
        assert!(matches!(
            model.predict(&[1.0]),
            Err(ModelError::InputDimension { .. })
        ));
    }
}
