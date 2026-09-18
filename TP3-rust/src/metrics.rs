use std::cmp::Ordering;

use thiserror::Error;

#[derive(Clone, Copy, Debug)]
pub struct RegressionMetrics {
    pub mse: f64,
    pub rmse: f64,
    pub r2: f64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MetricsError {
    #[error("predictions and targets must have the same non-zero length")]
    InvalidLengths,
    #[error("scores must be finite")]
    NonFiniteScore,
}

pub fn regression_metrics(
    predictions: &[f64],
    targets: &[f64],
) -> Result<RegressionMetrics, MetricsError> {
    validate_lengths(predictions.len(), targets.len())?;
    if predictions.iter().any(|value| !value.is_finite())
        || targets.iter().any(|value| !value.is_finite())
    {
        return Err(MetricsError::NonFiniteScore);
    }
    let count = predictions.len() as f64;
    let mse = predictions
        .iter()
        .zip(targets)
        .map(|(&prediction, &target)| (prediction - target).powi(2))
        .sum::<f64>()
        / count;
    let target_mean = targets.iter().sum::<f64>() / count;
    let total_variance = targets
        .iter()
        .map(|target| (target - target_mean).powi(2))
        .sum::<f64>();
    let residual = mse * count;
    let r2 = if total_variance <= f64::EPSILON {
        if residual <= f64::EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - residual / total_variance
    };
    Ok(RegressionMetrics {
        mse,
        rmse: mse.sqrt(),
        r2,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConfusionMatrix {
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
}

impl ConfusionMatrix {
    pub fn precision(self) -> f64 {
        safe_ratio(
            self.true_positives,
            self.true_positives + self.false_positives,
        )
    }

    pub fn recall(self) -> f64 {
        safe_ratio(
            self.true_positives,
            self.true_positives + self.false_negatives,
        )
    }

    pub fn specificity(self) -> f64 {
        safe_ratio(
            self.true_negatives,
            self.true_negatives + self.false_positives,
        )
    }

    pub fn accuracy(self) -> f64 {
        safe_ratio(
            self.true_positives + self.true_negatives,
            self.true_positives + self.true_negatives + self.false_positives + self.false_negatives,
        )
    }
}

#[derive(Clone, Debug)]
pub struct ThresholdMetrics {
    pub threshold: f64,
    pub confusion: ConfusionMatrix,
}

impl ThresholdMetrics {
    pub fn precision(&self) -> f64 {
        self.confusion.precision()
    }

    pub fn recall(&self) -> f64 {
        self.confusion.recall()
    }

    pub fn accuracy(&self) -> f64 {
        self.confusion.accuracy()
    }
}

pub fn confusion_matrix(
    scores: &[f64],
    labels: &[bool],
    threshold: f64,
) -> Result<ConfusionMatrix, MetricsError> {
    validate_lengths(scores.len(), labels.len())?;
    if !threshold.is_finite() || scores.iter().any(|score| !score.is_finite()) {
        return Err(MetricsError::NonFiniteScore);
    }
    let mut confusion = ConfusionMatrix::default();
    for (&score, &label) in scores.iter().zip(labels) {
        match (score >= threshold, label) {
            (true, true) => confusion.true_positives += 1,
            (true, false) => confusion.false_positives += 1,
            (false, false) => confusion.true_negatives += 1,
            (false, true) => confusion.false_negatives += 1,
        }
    }
    Ok(confusion)
}

pub fn threshold_sweep(
    scores: &[f64],
    labels: &[bool],
) -> Result<Vec<ThresholdMetrics>, MetricsError> {
    validate_lengths(scores.len(), labels.len())?;
    if scores.iter().any(|score| !score.is_finite()) {
        return Err(MetricsError::NonFiniteScore);
    }
    let mut thresholds = scores.to_vec();
    thresholds.sort_by(|left, right| left.total_cmp(right));
    thresholds.dedup_by(|left, right| left.total_cmp(right) == Ordering::Equal);

    use rayon::prelude::*;
    thresholds
        .into_par_iter()
        .map(|threshold| {
            Ok(ThresholdMetrics {
                threshold,
                confusion: confusion_matrix(scores, labels, threshold)?,
            })
        })
        .collect()
}

pub fn best_accuracy_threshold(sweep: &[ThresholdMetrics]) -> Option<&ThresholdMetrics> {
    sweep.iter().max_by(|left, right| {
        left.accuracy()
            .total_cmp(&right.accuracy())
            .then_with(|| left.recall().total_cmp(&right.recall()))
            .then_with(|| right.threshold.total_cmp(&left.threshold))
    })
}

fn validate_lengths(left: usize, right: usize) -> Result<(), MetricsError> {
    if left == 0 || left != right {
        Err(MetricsError::InvalidLengths)
    } else {
        Ok(())
    }
}

fn safe_ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn regression_metrics_match_known_values() {
        let metrics = regression_metrics(&[1.0, 3.0], &[0.0, 1.0]).unwrap();
        assert_abs_diff_eq!(metrics.mse, 2.5);
        assert_abs_diff_eq!(metrics.rmse, 2.5_f64.sqrt());
    }

    #[test]
    fn classification_metrics_match_confusion_counts() {
        let confusion =
            confusion_matrix(&[0.9, 0.8, 0.7, 0.1], &[true, false, true, false], 0.75).unwrap();
        assert_eq!(
            confusion,
            ConfusionMatrix {
                true_positives: 1,
                false_positives: 1,
                true_negatives: 1,
                false_negatives: 1,
            }
        );
        assert_abs_diff_eq!(confusion.precision(), 0.5);
        assert_abs_diff_eq!(confusion.recall(), 0.5);
        assert_abs_diff_eq!(confusion.accuracy(), 0.5);
    }

    #[test]
    fn accuracy_threshold_ties_prefer_recall_then_lower_threshold() {
        let sweep = threshold_sweep(&[0.9, 0.8, 0.7], &[true, false, true]).unwrap();
        let best = best_accuracy_threshold(&sweep).unwrap();
        assert_abs_diff_eq!(best.threshold, 0.7);
        assert_abs_diff_eq!(best.recall(), 1.0);
    }
}
