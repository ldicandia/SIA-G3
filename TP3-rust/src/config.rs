use std::{fs, path::Path};

use serde::Deserialize;
use thiserror::Error;

#[derive(Clone, Debug, Deserialize)]
pub struct AppConfig {
    pub split: SplitConfig,
    pub search: SearchConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SplitConfig {
    pub train_ratio: f64,
    pub validation_ratio: f64,
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

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("could not read configuration: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid TOML configuration: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("train and validation ratios must be positive and sum to less than one")]
    InvalidRatios,
    #[error("at least two stratification bins are required")]
    InvalidBins,
    #[error("search values must be finite and positive")]
    InvalidSearch,
}

impl AppConfig {
    pub fn from_path(path: &Path) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(&fs::read_to_string(path)?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(self.split.train_ratio > 0.0
            && self.split.validation_ratio > 0.0
            && self.split.train_ratio + self.split.validation_ratio < 1.0)
        {
            return Err(ConfigError::InvalidRatios);
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
        Ok(())
    }
}
