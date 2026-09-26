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

    /// Harmonic mean of precision and recall: `2TP / (2TP + FP + FN)`.
    pub fn f1(self) -> f64 {
        safe_ratio(
            2 * self.true_positives,
            2 * self.true_positives + self.false_positives + self.false_negatives,
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

    pub fn f1(&self) -> f64 {
        self.confusion.f1()
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

/// Threshold that maximizes F1 on the sweep; ties prefer higher recall (a
/// missed fraud is costlier than a manual review) and then the lower threshold.
pub fn best_f1_threshold(sweep: &[ThresholdMetrics]) -> Option<&ThresholdMetrics> {
    sweep.iter().max_by(|left, right| {
        left.f1()
            .total_cmp(&right.f1())
            .then_with(|| left.recall().total_cmp(&right.recall()))
            .then_with(|| right.threshold.total_cmp(&left.threshold))
    })
}

/// Area under the precision-recall curve as average precision:
/// `sum_k (R_k - R_{k-1}) * P_k`, walking scores from high to low and
/// treating tied scores as a single operating point. Unlike ROC-AUC it is
/// sensitive to the rare positive class.
pub fn average_precision(scores: &[f64], labels: &[bool]) -> Result<f64, MetricsError> {
    validate_lengths(scores.len(), labels.len())?;
    if scores.iter().any(|score| !score.is_finite()) {
        return Err(MetricsError::NonFiniteScore);
    }
    let positives = labels.iter().filter(|&&label| label).count();
    if positives == 0 {
        return Ok(0.0);
    }
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
    let (mut true_positives, mut seen, mut previous_recall, mut area) = (0usize, 0usize, 0.0, 0.0);
    let mut position = 0;
    while position < order.len() {
        let score = scores[order[position]];
        while position < order.len() && scores[order[position]] == score {
            true_positives += usize::from(labels[order[position]]);
            seen += 1;
            position += 1;
        }
        let recall = true_positives as f64 / positives as f64;
        let precision = true_positives as f64 / seen as f64;
        area += (recall - previous_recall) * precision;
        previous_recall = recall;
    }
    Ok(area)
}

/// Mean and population standard deviation.
pub fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    (mean, variance.sqrt())
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
    fn f1_and_average_precision_match_hand_computed_values() {
        let confusion =
            confusion_matrix(&[0.9, 0.8, 0.7, 0.1], &[true, false, true, false], 0.75).unwrap();
        assert_abs_diff_eq!(confusion.f1(), 0.5);
        // Ranking: T(0.9) F(0.8) T(0.7) F(0.1) -> AP = 0.5 * 1 + 0.5 * 2/3.
        let ap = average_precision(&[0.9, 0.8, 0.7, 0.1], &[true, false, true, false]).unwrap();
        assert_abs_diff_eq!(ap, 0.5 + 1.0 / 3.0, epsilon = 1e-12);
        let sweep = threshold_sweep(&[0.9, 0.8, 0.7, 0.1], &[true, false, true, false]).unwrap();
        let best = best_f1_threshold(&sweep).unwrap();
        assert_abs_diff_eq!(best.threshold, 0.7);
        assert_abs_diff_eq!(best.f1(), 0.8);
    }

    #[test]
    fn accuracy_threshold_ties_prefer_recall_then_lower_threshold() {
        let sweep = threshold_sweep(&[0.9, 0.8, 0.7], &[true, false, true]).unwrap();
        let best = best_accuracy_threshold(&sweep).unwrap();
        assert_abs_diff_eq!(best.threshold, 0.7);
        assert_abs_diff_eq!(best.recall(), 1.0);
    }
}
