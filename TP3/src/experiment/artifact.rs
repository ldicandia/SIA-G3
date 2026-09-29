//! Persisted fraud TinyModel: feature preprocessing, scaler, weights and threshold.

use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    config::FeatureConfig,
    data::{transform_raw_row, ScalerError, StandardScaler},
    model::{ModelError, SingleLayerPerceptron},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FraudModelArtifact {
    pub format_version: u32,
    /// Raw column order expected by `score_raw`.
    pub raw_feature_names: Vec<String>,
    pub features: FeatureConfig,
    pub scaler: StandardScaler,
    pub model: SingleLayerPerceptron,
    pub learning_rate: f64,
    pub epochs: usize,
    pub threshold: f64,
}

#[derive(Debug, Error)]
pub enum FraudArtifactError {
    #[error("could not access model artifact: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid model artifact: {0}")]
    Decode(#[from] toml::de::Error),
    #[error("could not encode model artifact: {0}")]
    Encode(#[from] toml::ser::Error),
    #[error("unsupported model artifact version {0}")]
    UnsupportedVersion(u32),
    #[error(transparent)]
    Scaler(#[from] ScalerError),
    #[error(transparent)]
    Model(#[from] ModelError),
}

impl FraudModelArtifact {
    pub fn save(&self, path: &Path) -> Result<(), FraudArtifactError> {
        fs::write(path, toml::to_string(self)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, FraudArtifactError> {
        let artifact: Self = toml::from_str(&fs::read_to_string(path)?)?;
        if artifact.format_version != 1 {
            return Err(FraudArtifactError::UnsupportedVersion(
                artifact.format_version,
            ));
        }
        Ok(artifact)
    }

    /// Fraud probability for one raw transaction (columns in `raw_feature_names` order).
    pub fn score_raw(&self, raw: &[f64]) -> Result<f64, FraudArtifactError> {
        let transformed = transform_raw_row(&self.raw_feature_names, raw, &self.features);
        let scaled = self.scaler.transform_row(&transformed)?;
        Ok(self.model.predict(&scaled)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{matrix::DenseMatrix, model::Activation};

    #[test]
    fn artifact_round_trip_preserves_scores() {
        let raw = DenseMatrix::from_rows(vec![vec![1.0, 10.0], vec![3.0, 1000.0]]).unwrap();
        let features = FeatureConfig {
            drop: vec![],
            log1p: vec!["b".into()],
        };
        let names = vec!["a".to_owned(), "b".to_owned()];
        let transformed = DenseMatrix::from_rows(
            (0..2)
                .map(|row| transform_raw_row(&names, raw.row(row).unwrap(), &features))
                .collect(),
        )
        .unwrap();
        let artifact = FraudModelArtifact {
            format_version: 1,
            raw_feature_names: names,
            features,
            scaler: StandardScaler::fit(&transformed, &[0, 1]).unwrap(),
            model: SingleLayerPerceptron::new(2, Activation::Sigmoid, 3).unwrap(),
            learning_rate: 0.01,
            epochs: 5,
            threshold: 0.5,
        };
        let expected = artifact.score_raw(raw.row(1).unwrap()).unwrap();
        let decoded: FraudModelArtifact =
            toml::from_str(&toml::to_string(&artifact).unwrap()).unwrap();
        assert_eq!(decoded.score_raw(raw.row(1).unwrap()).unwrap(), expected);
    }
}
