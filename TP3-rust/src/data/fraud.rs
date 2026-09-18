use std::{collections::HashSet, path::Path};

use serde::Deserialize;
use thiserror::Error;

use crate::matrix::{DenseMatrix, MatrixError};

pub const FEATURE_NAMES: [&str; 9] = [
    "timestamp",
    "amount_usd",
    "quantity_purchased",
    "session_duration_seconds",
    "days_since_last_purchase",
    "account_age_days",
    "device_screen_resolution",
    "time_since_last_login_s",
    "items_viewed_before_purchase",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FraudRecord {
    timestamp: i64,
    amount_usd: f64,
    quantity_purchased: u32,
    session_duration_seconds: f64,
    days_since_last_purchase: f64,
    account_age_days: u32,
    device_screen_resolution: u64,
    time_since_last_login_s: f64,
    items_viewed_before_purchase: u32,
    big_model_fraud_probability: f64,
    flagged_fraud: u8,
}

#[derive(Clone, Debug)]
pub struct FraudDataset {
    pub features: DenseMatrix,
    pub teacher_targets: Vec<f64>,
    pub fraud_labels: Vec<bool>,
}

impl FraudDataset {
    pub fn len(&self) -> usize {
        self.teacher_targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.teacher_targets.is_empty()
    }

    pub fn exact_duplicate_count(&self) -> usize {
        let mut rows = HashSet::with_capacity(self.len());
        let mut duplicates = 0;
        for index in 0..self.len() {
            let mut signature = Vec::with_capacity(self.features.cols() + 2);
            signature.extend(
                self.features
                    .row_unchecked(index)
                    .iter()
                    .map(|value| value.to_bits()),
            );
            signature.push(self.teacher_targets[index].to_bits());
            signature.push(self.fraud_labels[index] as u64);
            if !rows.insert(signature) {
                duplicates += 1;
            }
        }
        duplicates
    }
}

#[derive(Debug, Error)]
pub enum DataError {
    #[error("could not read CSV data: {0}")]
    Csv(#[from] csv::Error),
    #[error("dataset is empty")]
    Empty,
    #[error("row {row} contains a non-finite numeric value")]
    NonFinite { row: usize },
    #[error("row {row} has a BigModel probability outside [0, 1]")]
    InvalidProbability { row: usize },
    #[error("row {row} has a flagged_fraud value other than 0 or 1")]
    InvalidLabel { row: usize },
    #[error(transparent)]
    Matrix(#[from] MatrixError),
}

pub fn load_fraud_dataset(path: &Path) -> Result<FraudDataset, DataError> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut features = Vec::new();
    let mut teacher_targets = Vec::new();
    let mut fraud_labels = Vec::new();

    for (index, result) in reader.deserialize::<FraudRecord>().enumerate() {
        let row_number = index + 2;
        let record = result?;
        let row = [
            record.timestamp as f64,
            record.amount_usd,
            record.quantity_purchased as f64,
            record.session_duration_seconds,
            record.days_since_last_purchase,
            record.account_age_days as f64,
            record.device_screen_resolution as f64,
            record.time_since_last_login_s,
            record.items_viewed_before_purchase as f64,
        ];
        if row.iter().any(|value| !value.is_finite())
            || !record.big_model_fraud_probability.is_finite()
        {
            return Err(DataError::NonFinite { row: row_number });
        }
        if !(0.0..=1.0).contains(&record.big_model_fraud_probability) {
            return Err(DataError::InvalidProbability { row: row_number });
        }
        if record.flagged_fraud > 1 {
            return Err(DataError::InvalidLabel { row: row_number });
        }
        features.extend_from_slice(&row);
        teacher_targets.push(record.big_model_fraud_probability);
        fraud_labels.push(record.flagged_fraud == 1);
    }

    if teacher_targets.is_empty() {
        return Err(DataError::Empty);
    }
    Ok(FraudDataset {
        features: DenseMatrix::new(teacher_targets.len(), FEATURE_NAMES.len(), features)?,
        teacher_targets,
        fraud_labels,
    })
}
