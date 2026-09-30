//! Column-level feature preprocessing for the fraud dataset: drops, `log1p`
//! transforms and derived features built from the raw columns.

use std::f64::consts::TAU;

use serde::{Deserialize, Serialize};
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

/// Features constructed from the raw columns. Each one reads raw columns by
/// name, so timestamp-derived features work even when `timestamp` itself is
/// dropped. Ratios of heavy-tailed columns are heavy-tailed too, so they are
/// log-compressed with `ln(1 + x)`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DerivedFeature {
    /// `ln(1 + amount_usd / max(quantity_purchased, 1))`.
    AmountPerItem,
    /// `ln(1 + items_viewed / max(session_seconds / 60, 1 / 60))`.
    ItemsViewedPerMinute,
    /// `ln(1 + items_viewed / max(quantity_purchased, 1))`.
    ViewedPerPurchased,
    /// `ln(1 + amount_usd / (account_age_days + 1))`.
    AmountPerAccountDay,
    /// `ln(1 + amount_usd) * ln(1 + quantity_purchased)`: an explicit
    /// interaction, which a single neuron cannot form from its inputs.
    AmountXQuantity,
    /// `sin` and `cos` of `2 pi hour / 24` (fractional UTC hour).
    HourOfDay,
    /// 1 on Saturday and Sunday (UTC), 0 otherwise.
    Weekend,
}

impl DerivedFeature {
    pub const ALL: [Self; 7] = [
        Self::AmountPerItem,
        Self::ItemsViewedPerMinute,
        Self::ViewedPerPurchased,
        Self::AmountPerAccountDay,
        Self::AmountXQuantity,
        Self::HourOfDay,
        Self::Weekend,
    ];

    /// Configuration name (the serde spelling).
    pub fn name(self) -> &'static str {
        match self {
            Self::AmountPerItem => "amount_per_item",
            Self::ItemsViewedPerMinute => "items_viewed_per_minute",
            Self::ViewedPerPurchased => "viewed_per_purchased",
            Self::AmountPerAccountDay => "amount_per_account_day",
            Self::AmountXQuantity => "amount_x_quantity",
            Self::HourOfDay => "hour_of_day",
            Self::Weekend => "weekend",
        }
    }

    /// Names of the columns appended to the model inputs.
    pub fn column_names(self) -> &'static [&'static str] {
        match self {
            Self::AmountPerItem => &["log1p_amount_per_item"],
            Self::ItemsViewedPerMinute => &["log1p_items_viewed_per_minute"],
            Self::ViewedPerPurchased => &["log1p_viewed_per_purchased"],
            Self::AmountPerAccountDay => &["log1p_amount_per_account_day"],
            Self::AmountXQuantity => &["log1p_amount_x_log1p_quantity"],
            Self::HourOfDay => &["hour_sin", "hour_cos"],
            Self::Weekend => &["is_weekend"],
        }
    }

    /// Raw columns read by `compute`, in the order it expects them.
    fn sources(self) -> &'static [&'static str] {
        match self {
            Self::AmountPerItem | Self::AmountXQuantity => &["amount_usd", "quantity_purchased"],
            Self::ItemsViewedPerMinute => {
                &["items_viewed_before_purchase", "session_duration_seconds"]
            }
            Self::ViewedPerPurchased => &["items_viewed_before_purchase", "quantity_purchased"],
            Self::AmountPerAccountDay => &["amount_usd", "account_age_days"],
            Self::HourOfDay | Self::Weekend => &["timestamp"],
        }
    }

    fn compute(self, source: &[f64], output: &mut Vec<f64>) {
        match self {
            Self::AmountPerItem => output.push((source[0] / source[1].max(1.0)).ln_1p()),
            Self::ItemsViewedPerMinute => {
                output.push((source[0] / (source[1] / 60.0).max(1.0 / 60.0)).ln_1p())
            }
            Self::ViewedPerPurchased => output.push((source[0] / source[1].max(1.0)).ln_1p()),
            Self::AmountPerAccountDay => output.push((source[0] / (source[1] + 1.0)).ln_1p()),
            Self::AmountXQuantity => output.push(source[0].ln_1p() * source[1].ln_1p()),
            Self::HourOfDay => {
                let seconds = source[0] as i64;
                let hour = seconds.rem_euclid(86_400) as f64 / 3_600.0;
                let angle = TAU * hour / 24.0;
                output.push(angle.sin());
                output.push(angle.cos());
            }
            Self::Weekend => {
                // Same convention as `timestamp_profile.csv`: 1970-01-01 was a
                // Thursday and 0 = Monday, so 5 and 6 are the weekend.
                let days = (source[0] as i64).div_euclid(86_400);
                let weekday = (days + 3).rem_euclid(7);
                output.push(if weekday >= 5 { 1.0 } else { 0.0 });
            }
        }
    }
}

/// A derived feature with its raw source columns resolved to indices.
struct ResolvedDerived {
    feature: DerivedFeature,
    columns: Vec<usize>,
}

fn resolve_derived(
    names: &[String],
    config: &FeatureConfig,
) -> Result<Vec<ResolvedDerived>, FeatureError> {
    config
        .derived
        .iter()
        .map(|&feature| {
            let columns = feature
                .sources()
                .iter()
                .map(|source| {
                    names
                        .iter()
                        .position(|name| name == source)
                        .ok_or_else(|| FeatureError::UnknownColumn((*source).to_owned()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ResolvedDerived { feature, columns })
        })
        .collect()
}

fn append_derived(derived: &[ResolvedDerived], row: &[f64], output: &mut Vec<f64>) {
    let mut source = Vec::with_capacity(2);
    for resolved in derived {
        source.clear();
        source.extend(resolved.columns.iter().map(|&column| row[column]));
        resolved.feature.compute(&source, output);
    }
}

/// Applies the configured drops and `log1p` transforms, then appends the
/// derived columns (in `derived` order), returning a new dataset whose
/// `feature_names` describe the transformed columns. Always called on the
/// raw dataset.
pub fn apply_feature_config(
    dataset: &FraudDataset,
    config: &FeatureConfig,
) -> Result<FraudDataset, FeatureError> {
    for name in config.drop.iter().chain(&config.log1p) {
        if !dataset.feature_names.contains(name) {
            return Err(FeatureError::UnknownColumn(name.clone()));
        }
    }
    let derived = resolve_derived(&dataset.feature_names, config)?;
    let kept = dataset
        .feature_names
        .iter()
        .enumerate()
        .filter(|(_, name)| !config.drop.contains(name))
        .map(|(column, name)| (column, name, config.log1p.contains(name)))
        .collect::<Vec<_>>();
    let mut names = kept
        .iter()
        .map(|&(_, name, log)| {
            if log {
                format!("log1p_{name}")
            } else {
                name.clone()
            }
        })
        .collect::<Vec<_>>();
    names.extend(
        config
            .derived
            .iter()
            .flat_map(|feature| feature.column_names().iter().map(|name| (*name).to_owned())),
    );

    let mut values = Vec::with_capacity(dataset.len() * names.len());
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
        append_derived(&derived, source, &mut values);
    }
    Ok(FraudDataset {
        features: DenseMatrix::new(dataset.len(), names.len(), values)?,
        feature_names: names,
        teacher_targets: dataset.teacher_targets.clone(),
        fraud_labels: dataset.fraud_labels.clone(),
    })
}

/// Applies the same transformation to a single raw row (in `raw_names` order).
pub fn transform_raw_row(
    raw_names: &[String],
    row: &[f64],
    config: &FeatureConfig,
) -> Result<Vec<f64>, FeatureError> {
    let derived = resolve_derived(raw_names, config)?;
    let mut values = raw_names
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
        .collect::<Vec<_>>();
    append_derived(&derived, row, &mut values);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::data::FEATURE_NAMES;

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
            ..Default::default()
        };
        let transformed = apply_feature_config(&dataset, &config).unwrap();
        assert_eq!(transformed.feature_names, ["b", "log1p_c"]);
        assert_abs_diff_eq!(transformed.features.row(1).unwrap()[1], 7.0_f64.ln());
        assert_eq!(
            transform_raw_row(&dataset.feature_names, &[4.0, 5.0, 6.0], &config).unwrap(),
            transformed.features.row(1).unwrap()
        );
    }

    /// Two raw rows in `FEATURE_NAMES` order: timestamp, amount, quantity,
    /// session, days_since_last_purchase, account_age, resolution,
    /// time_since_last_login, items_viewed.
    fn raw_dataset() -> FraudDataset {
        FraudDataset {
            feature_names: FEATURE_NAMES
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
            features: DenseMatrix::from_rows(vec![
                vec![0.0, 99.0, 3.0, 120.0, 4.0, 9.0, 2.0, 5.0, 10.0],
                vec![
                    2.0 * 86_400.0 + 6.0 * 3_600.0,
                    9.0,
                    1.0,
                    5.0,
                    1.0,
                    1.0,
                    2.0,
                    5.0,
                    4.0,
                ],
            ])
            .unwrap(),
            teacher_targets: vec![0.1, 0.9],
            fraud_labels: vec![false, true],
        }
    }

    #[test]
    fn derived_features_are_appended_with_hand_computed_values() {
        let dataset = raw_dataset();
        let config = FeatureConfig {
            drop: vec!["timestamp".into()],
            log1p: vec![],
            derived: DerivedFeature::ALL.to_vec(),
        };
        let transformed = apply_feature_config(&dataset, &config).unwrap();
        let expected_names = FEATURE_NAMES[1..]
            .iter()
            .copied()
            .chain([
                "log1p_amount_per_item",
                "log1p_items_viewed_per_minute",
                "log1p_viewed_per_purchased",
                "log1p_amount_per_account_day",
                "log1p_amount_x_log1p_quantity",
                "hour_sin",
                "hour_cos",
                "is_weekend",
            ])
            .collect::<Vec<_>>();
        assert_eq!(transformed.feature_names, expected_names);

        // Row 0: timestamp 0 is Thursday 00:00 UTC.
        let row = transformed.features.row(0).unwrap();
        let derived = &row[8..];
        assert_abs_diff_eq!(derived[0], (1.0 + 99.0 / 3.0_f64).ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(derived[1], (1.0 + 10.0 / 2.0_f64).ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(derived[2], (1.0 + 10.0 / 3.0_f64).ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(derived[3], (1.0 + 99.0 / 10.0_f64).ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(derived[4], 100.0_f64.ln() * 4.0_f64.ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(derived[5], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(derived[6], 1.0, epsilon = 1e-12);
        assert_eq!(derived[7], 0.0);

        // Row 1: Saturday 06:00 UTC. A 5-second session is 1/12 minute (the
        // 1/60-minute floor only applies below one second).
        let row = transformed.features.row(1).unwrap();
        let derived = &row[8..];
        assert_abs_diff_eq!(derived[0], 10.0_f64.ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(derived[1], (1.0 + 4.0 * 12.0_f64).ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(derived[3], (1.0 + 9.0 / 2.0_f64).ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(derived[5], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(derived[6], 0.0, epsilon = 1e-12);
        assert_eq!(derived[7], 1.0);

        for index in 0..2 {
            assert_eq!(
                transform_raw_row(
                    &dataset.feature_names,
                    dataset.features.row(index).unwrap(),
                    &config
                )
                .unwrap(),
                transformed.features.row(index).unwrap()
            );
        }
    }

    #[test]
    fn derived_feature_requires_its_source_column() {
        let dataset = FraudDataset {
            feature_names: vec!["amount_usd".into()],
            features: DenseMatrix::from_rows(vec![vec![1.0]]).unwrap(),
            teacher_targets: vec![0.5],
            fraud_labels: vec![false],
        };
        let config = FeatureConfig {
            derived: vec![DerivedFeature::AmountPerItem],
            ..Default::default()
        };
        assert!(matches!(
            apply_feature_config(&dataset, &config),
            Err(FeatureError::UnknownColumn(column)) if column == "quantity_purchased"
        ));
    }

    #[test]
    fn derived_list_defaults_to_empty_and_is_not_serialized_when_empty() {
        let config: FeatureConfig =
            toml::from_str("drop = [\"timestamp\"]\nlog1p = [\"amount_usd\"]\n").unwrap();
        assert!(config.derived.is_empty());
        let encoded = toml::to_string(&config).unwrap();
        assert!(!encoded.contains("derived"), "{encoded}");

        let with_derived = FeatureConfig {
            derived: vec![DerivedFeature::HourOfDay, DerivedFeature::AmountXQuantity],
            ..config
        };
        let encoded = toml::to_string(&with_derived).unwrap();
        assert!(encoded.contains("hour_of_day"), "{encoded}");
        let decoded: FeatureConfig = toml::from_str(&encoded).unwrap();
        assert_eq!(decoded, with_derived);
    }
}
