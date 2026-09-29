mod activation;
mod multi_layer_perceptron;
mod perceptron;

use thiserror::Error;

pub use activation::*;
pub use multi_layer_perceptron::*;
pub use perceptron::*;

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
