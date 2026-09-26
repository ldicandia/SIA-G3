use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::{Activation, Initialization};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    #[default]
    Sgd,
    Momentum,
    Adam,
}

/// One fully specified training run. Every field is stored inside the
/// persisted model so a run can be audited or resumed with the same settings.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CandidateConfig {
    pub name: String,
    /// Study axis this candidate belongs to (e.g. `learning_rate`,
    /// `architecture`, `optimizer`), used to group the comparison tables.
    #[serde(default)]
    pub axis: String,
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
    #[serde(default)]
    pub initialization: Initialization,
    /// Decoupled L2 weight decay applied to weights (not biases) each step.
    #[serde(default)]
    pub weight_decay: f64,
    /// Inverted-dropout probability for hidden units during training.
    #[serde(default)]
    pub dropout: f64,
    /// Step decay: the learning rate is multiplied by `lr_decay_factor` every
    /// `lr_decay_every` epochs (0 keeps it constant).
    #[serde(default = "default_decay_factor")]
    pub lr_decay_factor: f64,
    #[serde(default)]
    pub lr_decay_every: usize,
    /// Data augmentation: random translation of up to this many pixels.
    #[serde(default)]
    pub augment_shift: f64,
    /// Data augmentation: random rotation of up to this many degrees.
    #[serde(default)]
    pub augment_rotation: f64,
}

impl Default for CandidateConfig {
    fn default() -> Self {
        Self {
            name: "candidate".into(),
            axis: String::new(),
            topology: vec![784, 64, 10],
            hidden_activation: Activation::Relu,
            optimizer: OptimizerKind::Adam,
            learning_rate: 0.001,
            batch_size: 64,
            max_epochs: 20,
            patience: 4,
            min_delta: 1e-5,
            seed: 42,
            momentum: default_momentum(),
            beta1: default_beta1(),
            beta2: default_beta2(),
            epsilon: default_epsilon(),
            initialization: Initialization::Xavier,
            weight_decay: 0.0,
            dropout: 0.0,
            lr_decay_factor: default_decay_factor(),
            lr_decay_every: 0,
            augment_shift: 0.0,
            augment_rotation: 0.0,
        }
    }
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
        let valid_regularization = self.weight_decay.is_finite()
            && self.weight_decay >= 0.0
            && self.dropout.is_finite()
            && (0.0..1.0).contains(&self.dropout)
            && self.lr_decay_factor.is_finite()
            && self.lr_decay_factor > 0.0
            && self.lr_decay_factor <= 1.0
            && self.augment_shift.is_finite()
            && self.augment_shift >= 0.0
            && self.augment_rotation.is_finite()
            && self.augment_rotation >= 0.0;
        if positive && valid_topology && valid_optimizer && valid_regularization {
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

    /// Learning rate used during a 1-based global epoch.
    pub fn learning_rate_at(&self, epoch: usize) -> f64 {
        if self.lr_decay_every == 0 {
            return self.learning_rate;
        }
        let decays = (epoch.saturating_sub(1) / self.lr_decay_every) as i32;
        self.learning_rate * self.lr_decay_factor.powi(decays)
    }

    pub fn augments(&self) -> bool {
        self.augment_shift > 0.0 || self.augment_rotation > 0.0
    }

    /// Axis label used in reports; candidates without one are grouped as "other".
    pub fn axis_label(&self) -> &str {
        if self.axis.trim().is_empty() {
            "other"
        } else {
            &self.axis
        }
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

fn default_decay_factor() -> f64 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(name: &str) -> CandidateConfig {
        CandidateConfig {
            name: name.into(),
            topology: vec![2, 3, 10],
            batch_size: 4,
            max_epochs: 10,
            patience: 3,
            min_delta: 1e-6,
            ..CandidateConfig::default()
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

    #[test]
    fn step_decay_halves_the_rate_every_period() {
        let config = CandidateConfig {
            learning_rate: 0.1,
            lr_decay_factor: 0.5,
            lr_decay_every: 10,
            ..candidate("decay")
        };
        assert_eq!(config.learning_rate_at(1), 0.1);
        assert_eq!(config.learning_rate_at(10), 0.1);
        assert_eq!(config.learning_rate_at(11), 0.05);
        assert_eq!(config.learning_rate_at(21), 0.025);
    }

    #[test]
    fn legacy_candidates_without_new_fields_still_parse() {
        let text = r#"
            name = "legacy"
            topology = [784, 64, 10]
            hidden_activation = "relu"
            optimizer = "adam"
            learning_rate = 0.001
            batch_size = 64
            max_epochs = 5
            patience = 2
            min_delta = 0.0
            seed = 1
        "#;
        let candidate: CandidateConfig = toml::from_str(text).unwrap();
        assert_eq!(candidate.initialization, Initialization::Xavier);
        assert_eq!(candidate.dropout, 0.0);
        assert_eq!(candidate.learning_rate_at(100), 0.001);
    }
}
