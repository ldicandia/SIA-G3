use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Linear,
    Sigmoid,
    Tanh,
    Relu,
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
            Self::Relu => value.max(0.0),
        }
    }

    pub fn derivative_from_output(self, output: f64) -> f64 {
        match self {
            Self::Linear => 1.0,
            Self::Sigmoid => output * (1.0 - output),
            Self::Tanh => 1.0 - output.powi(2),
            Self::Relu => f64::from(output > 0.0),
        }
    }
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
}
