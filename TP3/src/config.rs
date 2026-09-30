use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::data::{DerivedFeature, FEATURE_NAMES};

#[derive(Clone, Debug, Deserialize)]
pub struct AppConfig {
    pub split: SplitConfig,
    pub search: SearchConfig,
    #[serde(default)]
    pub features: FeatureConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SplitConfig {
    /// Fraction of the dataset held out and consulted only once, at the end.
    pub test_ratio: f64,
    /// Number of cross-validation folds built over the remaining development rows.
    pub folds: usize,
    pub seed: u64,
    pub stratification_bins: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SearchConfig {
    pub learning_rates: Vec<f64>,
    pub max_epochs: usize,
    pub patience: usize,
    pub min_delta: f64,
    pub initialization_seed: u64,
}

/// Column-level preprocessing applied before scaling. Stored with the fraud
/// model so the same transformation is reproduced when scoring new data.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct FeatureConfig {
    /// Raw columns removed from the model inputs.
    #[serde(default)]
    pub drop: Vec<String>,
    /// Raw columns replaced by `ln(1 + x)` to compress heavy right tails.
    #[serde(default)]
    pub log1p: Vec<String>,
    /// Constructed features appended after the kept raw columns. Omitted
    /// from serialized artifacts when empty, so older models stay readable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derived: Vec<DerivedFeature>,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("could not read configuration: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid TOML configuration: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("test_ratio must be in (0, 1) and folds must be at least 2")]
    InvalidSplit,
    #[error("at least two stratification bins are required")]
    InvalidBins,
    #[error("search values must be finite and positive")]
    InvalidSearch,
    #[error("unknown or repeated feature '{0}' in [features]")]
    InvalidFeature(String),
}

impl AppConfig {
    pub fn from_path(path: &Path) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(&fs::read_to_string(path)?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(self.split.test_ratio > 0.0 && self.split.test_ratio < 1.0) || self.split.folds < 2 {
            return Err(ConfigError::InvalidSplit);
        }
        if self.split.stratification_bins < 2 {
            return Err(ConfigError::InvalidBins);
        }
        if self.search.learning_rates.is_empty()
            || self
                .search
                .learning_rates
                .iter()
                .any(|rate| !rate.is_finite() || *rate <= 0.0)
            || self.search.max_epochs == 0
            || self.search.patience == 0
            || !self.search.min_delta.is_finite()
            || self.search.min_delta < 0.0
        {
            return Err(ConfigError::InvalidSearch);
        }
        self.features.validate()
    }
}

impl FeatureConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut seen = std::collections::HashSet::new();
        for name in self.drop.iter().chain(&self.log1p) {
            if !FEATURE_NAMES.contains(&name.as_str()) || !seen.insert(name) {
                return Err(ConfigError::InvalidFeature(name.clone()));
            }
        }
        let mut derived = std::collections::HashSet::new();
        for feature in &self.derived {
            if !derived.insert(feature) {
                return Err(ConfigError::InvalidFeature(feature.name().into()));
            }
        }
        if self.drop.len() == FEATURE_NAMES.len() && self.derived.is_empty() {
            return Err(ConfigError::InvalidFeature("all features dropped".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_config_rejects_unknown_and_repeated_columns() {
        let unknown = FeatureConfig {
            drop: vec!["nope".into()],
            ..Default::default()
        };
        assert!(unknown.validate().is_err());
        let repeated = FeatureConfig {
            drop: vec!["timestamp".into()],
            log1p: vec!["timestamp".into()],
            ..Default::default()
        };
        assert!(repeated.validate().is_err());
        let repeated_derived = FeatureConfig {
            derived: vec![DerivedFeature::Weekend, DerivedFeature::Weekend],
            ..Default::default()
        };
        assert!(repeated_derived.validate().is_err());
    }

    #[test]
    fn derived_features_alone_are_a_valid_input_set() {
        let all_dropped = FeatureConfig {
            drop: FEATURE_NAMES
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
            ..Default::default()
        };
        assert!(all_dropped.validate().is_err());
        let only_derived = FeatureConfig {
            derived: vec![DerivedFeature::HourOfDay],
            ..all_dropped
        };
        assert!(only_derived.validate().is_ok());
    }
}
