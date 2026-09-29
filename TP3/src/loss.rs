use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LossError {
    #[error("predictions and targets must have the same non-zero length")]
    InvalidLengths,
}

pub trait Loss {
    fn sample(&self, prediction: f64, target: f64) -> f64;
    fn derivative(&self, prediction: f64, target: f64) -> f64;

    fn mean(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, LossError> {
        if predictions.is_empty() || predictions.len() != targets.len() {
            return Err(LossError::InvalidLengths);
        }
        Ok(predictions
            .iter()
            .zip(targets)
            .map(|(&prediction, &target)| self.sample(prediction, target))
            .sum::<f64>()
            / predictions.len() as f64)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MeanSquaredError;

impl Loss for MeanSquaredError {
    fn sample(&self, prediction: f64, target: f64) -> f64 {
        (prediction - target).powi(2)
    }

    fn derivative(&self, prediction: f64, target: f64) -> f64 {
        2.0 * (prediction - target)
    }
}

pub fn categorical_cross_entropy(probabilities: &[f64], target: usize) -> Result<f64, LossError> {
    let probability = probabilities.get(target).ok_or(LossError::InvalidLengths)?;
    if probabilities.is_empty() {
        return Err(LossError::InvalidLengths);
    }
    Ok(-probability.max(f64::MIN_POSITIVE).ln())
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn mse_value_and_derivative_are_correct() {
        let loss = MeanSquaredError;
        assert_abs_diff_eq!(loss.mean(&[1.0, 3.0], &[0.0, 1.0]).unwrap(), 2.5);
        assert_abs_diff_eq!(loss.derivative(3.0, 1.0), 4.0);
    }

    #[test]
    fn mse_rejects_empty_or_mismatched_inputs() {
        let loss = MeanSquaredError;
        assert_eq!(loss.mean(&[], &[]), Err(LossError::InvalidLengths));
        assert_eq!(loss.mean(&[1.0], &[]), Err(LossError::InvalidLengths));
    }

    #[test]
    fn cross_entropy_uses_the_target_probability() {
        assert_abs_diff_eq!(
            categorical_cross_entropy(&[0.1, 0.7, 0.2], 1).unwrap(),
            -0.7_f64.ln()
        );
        assert_eq!(
            categorical_cross_entropy(&[0.5, 0.5], 2),
            Err(LossError::InvalidLengths)
        );
    }
}
