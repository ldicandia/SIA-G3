use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::MultilayerPerceptron;

use super::config::CandidateConfig;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DigitModelArtifact {
    pub format_version: u32,
    pub exercise: String,
    pub candidate: CandidateConfig,
    pub selected_epoch: usize,
    pub selection_validation_loss: f64,
    pub selection_validation_accuracy: f64,
    pub model: MultilayerPerceptron,
}

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("could not access model artifact: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid model artifact: {0}")]
    Decode(#[from] toml::de::Error),
    #[error("could not encode model artifact: {0}")]
    Encode(#[from] toml::ser::Error),
    #[error("unsupported model artifact version {0}")]
    UnsupportedVersion(u32),
}

impl DigitModelArtifact {
    pub fn save(&self, path: &Path) -> Result<(), ArtifactError> {
        fs::write(path, toml::to_string(self)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, ArtifactError> {
        let artifact: Self = toml::from_str(&fs::read_to_string(path)?)?;
        if artifact.format_version != 1 {
            return Err(ArtifactError::UnsupportedVersion(artifact.format_version));
        }
        Ok(artifact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        digits::config::OptimizerKind,
        model::{Activation, MultilayerPerceptron},
    };

    #[test]
    fn artifact_round_trip_preserves_predictions() {
        let candidate = CandidateConfig {
            name: "round-trip".into(),
            topology: vec![2, 3, 10],
            hidden_activation: Activation::Relu,
            optimizer: OptimizerKind::Sgd,
            learning_rate: 0.1,
            batch_size: 1,
            max_epochs: 1,
            patience: 1,
            min_delta: 0.0,
            seed: 4,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let model = MultilayerPerceptron::new(
            &candidate.topology,
            &[Activation::Relu, Activation::Linear],
            candidate.seed,
        )
        .unwrap();
        let expected = model.predict_probabilities(&[0.2, 0.8]).unwrap();
        let artifact = DigitModelArtifact {
            format_version: 1,
            exercise: "test".into(),
            candidate,
            selected_epoch: 1,
            selection_validation_loss: 1.0,
            selection_validation_accuracy: 0.5,
            model,
        };
        let encoded = toml::to_string(&artifact).unwrap();
        let decoded: DigitModelArtifact = toml::from_str(&encoded).unwrap();
        assert_eq!(
            decoded.model.predict_probabilities(&[0.2, 0.8]).unwrap(),
            expected
        );
    }
}
