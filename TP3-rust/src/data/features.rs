//! Column-level feature preprocessing for the fraud dataset.

use thiserror::Error;

use crate::{
    config::FeatureConfig,
    matrix::{DenseMatrix, MatrixError},
};

use super::FraudDataset;

#[derive(Debug, Error)]
pub enum FeatureError {
    #[error("column '{0}' is not present in the dataset")]
    UnknownColumn(String),
    #[error("log1p requires values greater than -1 in column '{0}'")]
    InvalidLog(String),
    #[error(transparent)]
    Matrix(#[from] MatrixError),
}

/// Applies the configured drops and `log1p` transforms, returning a new
/// dataset whose `feature_names` describe the transformed columns.
pub fn apply_feature_config(
    dataset: &FraudDataset,
    config: &FeatureConfig,
) -> Result<FraudDataset, FeatureError> {
    for name in config.drop.iter().chain(&config.log1p) {
        if !dataset.feature_names.contains(name) {
            return Err(FeatureError::UnknownColumn(name.clone()));
        }
    }
    let kept = dataset
        .feature_names
        .iter()
        .enumerate()
        .filter(|(_, name)| !config.drop.contains(name))
        .map(|(column, name)| (column, name, config.log1p.contains(name)))
        .collect::<Vec<_>>();

    let mut values = Vec::with_capacity(dataset.len() * kept.len());
    for row in 0..dataset.len() {
        let source = dataset.features.row_unchecked(row);
        for &(column, name, log) in &kept {
            let value = source[column];
            if log {
                if value <= -1.0 {
                    return Err(FeatureError::InvalidLog(name.clone()));
                }
                values.push(value.ln_1p());
            } else {
                values.push(value);
            }
        }
    }
    Ok(FraudDataset {
        feature_names: kept
            .iter()
            .map(|&(_, name, log)| {
                if log {
                    format!("log1p_{name}")
                } else {
                    name.clone()
                }
            })
            .collect(),
        features: DenseMatrix::new(dataset.len(), kept.len(), values)?,
        teacher_targets: dataset.teacher_targets.clone(),
        fraud_labels: dataset.fraud_labels.clone(),
    })
}

/// Applies the same transformation to a single raw row (in `FEATURE_NAMES` order).
pub fn transform_raw_row(raw_names: &[String], row: &[f64], config: &FeatureConfig) -> Vec<f64> {
    raw_names
        .iter()
        .zip(row)
        .filter(|(name, _)| !config.drop.contains(name))
        .map(|(name, &value)| {
            if config.log1p.contains(name) {
                value.ln_1p()
            } else {
                value
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn drops_and_log_transforms_selected_columns() {
        let dataset = FraudDataset {
            feature_names: vec!["a".into(), "b".into(), "c".into()],
            features: DenseMatrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])
                .unwrap(),
            teacher_targets: vec![0.1, 0.9],
            fraud_labels: vec![false, true],
        };
        let config = FeatureConfig {
            drop: vec!["a".into()],
            log1p: vec!["c".into()],
        };
        let transformed = apply_feature_config(&dataset, &config).unwrap();
        assert_eq!(transformed.feature_names, ["b", "log1p_c"]);
        assert_abs_diff_eq!(transformed.features.row(1).unwrap()[1], 7.0_f64.ln());
        assert_eq!(
            transform_raw_row(&dataset.feature_names, &[4.0, 5.0, 6.0], &config),
            transformed.features.row(1).unwrap()
        );
    }
}
