use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::Activation;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    Sgd,
    Momentum,
    Adam,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CandidateConfig {
    pub name: String,
    pub topology: Vec<usize>,
    pub hidden_activation: Activation,
    pub optimizer: OptimizerKind,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub patience: usize,
    pub min_delta: f64,
    pub seed: u64,
    #[serde(default = "default_momentum")]
    pub momentum: f64,
    #[serde(default = "default_beta1")]
    pub beta1: f64,
    #[serde(default = "default_beta2")]
    pub beta2: f64,
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DigitStudyConfig {
    pub validation_ratio: f64,
    pub split_seed: u64,
    pub candidates: Vec<CandidateConfig>,
}

#[derive(Debug, Error)]
pub enum DigitConfigError {
    #[error("could not read configuration: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid TOML configuration: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("validation_ratio must be finite and in (0, 1)")]
    InvalidValidationRatio,
    #[error("at least one candidate is required")]
    MissingCandidates,
    #[error("candidate names must be non-empty and unique")]
    InvalidCandidateNames,
    #[error("candidate '{0}' has invalid hyperparameters")]
    InvalidCandidate(String),
}

impl DigitStudyConfig {
    pub fn from_path(path: &Path) -> Result<Self, DigitConfigError> {
        let config: Self = toml::from_str(&fs::read_to_string(path)?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), DigitConfigError> {
        if !self.validation_ratio.is_finite() || !(0.0..1.0).contains(&self.validation_ratio) {
            return Err(DigitConfigError::InvalidValidationRatio);
        }
        if self.candidates.is_empty() {
            return Err(DigitConfigError::MissingCandidates);
        }
        let mut names = std::collections::HashSet::new();
        if self
            .candidates
            .iter()
            .any(|candidate| candidate.name.trim().is_empty() || !names.insert(&candidate.name))
        {
            return Err(DigitConfigError::InvalidCandidateNames);
        }
        for candidate in &self.candidates {
            candidate.validate()?;
        }
        Ok(())
    }
}

impl CandidateConfig {
    pub fn validate(&self) -> Result<(), DigitConfigError> {
        let positive = self.learning_rate.is_finite()
            && self.learning_rate > 0.0
            && self.batch_size > 0
            && self.max_epochs > 0
            && self.patience > 0
            && self.min_delta.is_finite()
            && self.min_delta >= 0.0;
        let valid_topology = self.topology.len() >= 3
            && self.topology.iter().all(|&size| size > 0)
            && self.topology.last() == Some(&10);
        let valid_optimizer = match self.optimizer {
            OptimizerKind::Sgd => true,
            OptimizerKind::Momentum => {
                self.momentum.is_finite() && (0.0..1.0).contains(&self.momentum)
            }
            OptimizerKind::Adam => {
                self.beta1.is_finite()
                    && self.beta2.is_finite()
                    && (0.0..1.0).contains(&self.beta1)
                    && (0.0..1.0).contains(&self.beta2)
                    && self.epsilon.is_finite()
                    && self.epsilon > 0.0
            }
        };
        if positive && valid_topology && valid_optimizer {
            Ok(())
        } else {
            Err(DigitConfigError::InvalidCandidate(self.name.clone()))
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.topology
            .windows(2)
            .map(|sizes| sizes[0] * sizes[1] + sizes[1])
            .sum()
    }
}

fn default_momentum() -> f64 {
    0.9
}

fn default_beta1() -> f64 {
    0.9
}

fn default_beta2() -> f64 {
    0.999
}

fn default_epsilon() -> f64 {
    1e-8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(name: &str) -> CandidateConfig {
        CandidateConfig {
            name: name.into(),
            topology: vec![2, 3, 10],
            hidden_activation: Activation::Relu,
            optimizer: OptimizerKind::Adam,
            learning_rate: 0.001,
            batch_size: 4,
            max_epochs: 10,
            patience: 3,
            min_delta: 1e-6,
            seed: 42,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    #[test]
    fn candidate_parameter_count_includes_biases() {
        assert_eq!(candidate("valid").parameter_count(), 49);
    }

    #[test]
    fn duplicate_candidate_names_are_rejected() {
        let config = DigitStudyConfig {
            validation_ratio: 0.2,
            split_seed: 42,
            candidates: vec![candidate("same"), candidate("same")],
        };
        assert!(matches!(
            config.validate(),
            Err(DigitConfigError::InvalidCandidateNames)
        ));
    }
}
