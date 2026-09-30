//! Probability calibration for binary scores: calibration metrics (Brier
//! score, expected calibration error, reliability bins) and two
//! recalibrators (Platt scaling and isotonic regression). Pure functions with
//! no I/O; the experiment that uses them lives in `experiment::calibration`.
//!
//! A score `s` is calibrated when `P(y = 1 | score = s) ≈ s`: among all rows
//! scored 0.3, about 30% are positive.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum CalibrationError {
    #[error("probabilities and labels must have the same non-zero length")]
    InvalidLengths,
    #[error("scores must be finite and in [0, 1]")]
    InvalidScore,
    #[error("at least one reliability bin is required")]
    InvalidBins,
    #[error("calibration needs both positive and negative labels")]
    SingleClass,
    #[error("Platt scaling produced non-finite parameters")]
    NonFinite,
}

fn validate(probabilities: &[f64], labels: &[bool]) -> Result<(), CalibrationError> {
    if probabilities.is_empty() || probabilities.len() != labels.len() {
        return Err(CalibrationError::InvalidLengths);
    }
    if probabilities
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(CalibrationError::InvalidScore);
    }
    Ok(())
}

fn require_both_classes(labels: &[bool]) -> Result<(), CalibrationError> {
    let positives = labels.iter().filter(|&&label| label).count();
    if positives == 0 || positives == labels.len() {
        return Err(CalibrationError::SingleClass);
    }
    Ok(())
}

/// Mean squared error between probabilities and 0/1 outcomes.
pub fn brier_score(probabilities: &[f64], labels: &[bool]) -> Result<f64, CalibrationError> {
    validate(probabilities, labels)?;
    Ok(probabilities
        .iter()
        .zip(labels)
        .map(|(&p, &y)| (p - f64::from(u8::from(y))).powi(2))
        .sum::<f64>()
        / probabilities.len() as f64)
}

/// One equal-width bin of a reliability diagram. `mean_predicted` and
/// `observed_rate` are `None` for empty bins.
#[derive(Clone, Debug, PartialEq)]
pub struct ReliabilityBin {
    pub index: usize,
    pub low: f64,
    pub high: f64,
    pub count: usize,
    pub mean_predicted: Option<f64>,
    pub observed_rate: Option<f64>,
}

/// Equal-width bins on [0, 1]; `p = 1.0` falls in the last bin.
pub fn reliability_bins(
    probabilities: &[f64],
    labels: &[bool],
    bins: usize,
) -> Result<Vec<ReliabilityBin>, CalibrationError> {
    if bins == 0 {
        return Err(CalibrationError::InvalidBins);
    }
    validate(probabilities, labels)?;
    let mut sums = vec![(0usize, 0.0, 0usize); bins];
    for (&p, &y) in probabilities.iter().zip(labels) {
        let index = ((p * bins as f64).floor() as usize).min(bins - 1);
        sums[index].0 += 1;
        sums[index].1 += p;
        sums[index].2 += usize::from(y);
    }
    Ok(sums
        .into_iter()
        .enumerate()
        .map(|(index, (count, total, positives))| ReliabilityBin {
            index,
            low: index as f64 / bins as f64,
            high: (index + 1) as f64 / bins as f64,
            count,
            mean_predicted: (count > 0).then(|| total / count as f64),
            observed_rate: (count > 0).then(|| positives as f64 / count as f64),
        })
        .collect())
}

/// `ECE = sum_b (n_b / N) * |mean_predicted_b - observed_rate_b|` over the
/// non-empty equal-width bins.
pub fn expected_calibration_error(
    probabilities: &[f64],
    labels: &[bool],
    bins: usize,
) -> Result<f64, CalibrationError> {
    let total = probabilities.len() as f64;
    Ok(reliability_bins(probabilities, labels, bins)?
        .iter()
        .filter_map(|bin| {
            let gap = (bin.mean_predicted? - bin.observed_rate?).abs();
            Some(bin.count as f64 / total * gap)
        })
        .sum())
}

const LOGIT_CLAMP: f64 = 1e-6;

fn logit(score: f64) -> f64 {
    let p = score.clamp(LOGIT_CLAMP, 1.0 - LOGIT_CLAMP);
    (p / (1.0 - p)).ln()
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

/// `ln(1 + e^x)` without overflow.
fn softplus(value: f64) -> f64 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

/// Platt scaling on the logit of a probability score:
/// `p = sigmoid(a * logit(s) + b)`.
///
/// The TinyModel score is `s = sigmoid(w·x + c)`, so `logit(s) = w·x + c`
/// and the calibrated probability is `sigmoid((a w)·x + (a c + b))`: Platt
/// scaling only rescales the neuron's weights and bias. The calibrated
/// TinyModel is still one perceptron with the same 9 parameters, and because
/// `a > 0` it preserves the ranking (and therefore average precision).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlattScaling {
    pub a: f64,
    pub b: f64,
}

impl PlattScaling {
    /// Newton-Raphson on the log loss with Platt's smoothed targets
    /// (`(N+ + 1) / (N+ + 2)` for positives, `1 / (N- + 2)` for negatives),
    /// starting from the identity `a = 1, b = 0`, with step halving whenever
    /// a full Newton step does not reduce the loss.
    pub fn fit(scores: &[f64], labels: &[bool]) -> Result<Self, CalibrationError> {
        validate(scores, labels)?;
        require_both_classes(labels)?;
        let positives = labels.iter().filter(|&&label| label).count() as f64;
        let negatives = labels.len() as f64 - positives;
        let high = (positives + 1.0) / (positives + 2.0);
        let low = 1.0 / (negatives + 2.0);
        let inputs = scores.iter().map(|&s| logit(s)).collect::<Vec<_>>();
        let targets = labels
            .iter()
            .map(|&label| if label { high } else { low })
            .collect::<Vec<_>>();
        let loss = |a: f64, b: f64| {
            inputs
                .iter()
                .zip(&targets)
                .map(|(&z, &t)| {
                    let f = a * z + b;
                    t * softplus(-f) + (1.0 - t) * softplus(f)
                })
                .sum::<f64>()
        };

        let (mut a, mut b) = (1.0, 0.0);
        let mut current = loss(a, b);
        for _ in 0..100 {
            let (mut ga, mut gb, mut haa, mut hab, mut hbb) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for (&z, &t) in inputs.iter().zip(&targets) {
                let p = sigmoid(a * z + b);
                let residual = p - t;
                let weight = p * (1.0 - p);
                ga += residual * z;
                gb += residual;
                haa += weight * z * z;
                hab += weight * z;
                hbb += weight;
            }
            // Tiny ridge keeps the 2x2 system solvable on saturated data.
            haa += 1e-12;
            hbb += 1e-12;
            let determinant = haa * hbb - hab * hab;
            if !determinant.is_finite() || determinant.abs() <= f64::MIN_POSITIVE {
                break;
            }
            let step_a = (hbb * ga - hab * gb) / determinant;
            let step_b = (haa * gb - hab * ga) / determinant;
            let mut scale = 1.0;
            let mut accepted = false;
            while scale >= 1e-10 {
                let (next_a, next_b) = (a - scale * step_a, b - scale * step_b);
                let next = loss(next_a, next_b);
                if next.is_finite() && next <= current + 1e-12 * current.abs() {
                    a = next_a;
                    b = next_b;
                    current = next;
                    accepted = true;
                    break;
                }
                scale *= 0.5;
            }
            let step_norm = scale * (step_a * step_a + step_b * step_b).sqrt();
            if !accepted || step_norm < 1e-10 {
                break;
            }
        }
        if !a.is_finite() || !b.is_finite() {
            return Err(CalibrationError::NonFinite);
        }
        Ok(Self { a, b })
    }

    pub fn apply(&self, score: f64) -> f64 {
        sigmoid(self.a * logit(score) + self.b)
    }
}

/// One constant piece of the isotonic step function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsotonicBlock {
    pub min_score: f64,
    pub max_score: f64,
    pub value: f64,
}

/// Non-decreasing step function fitted with pool-adjacent-violators.
#[derive(Clone, Debug, PartialEq)]
pub struct IsotonicRegression {
    pub blocks: Vec<IsotonicBlock>,
}

impl IsotonicRegression {
    /// Pool-adjacent-violators on the scores sorted ascending. Tied scores
    /// start as one block; whenever a block's mean label is not above the
    /// previous block's, the two are merged into their weighted mean (equal
    /// neighbours are merged too, so every block has a distinct value).
    pub fn fit(scores: &[f64], labels: &[bool]) -> Result<Self, CalibrationError> {
        validate(scores, labels)?;
        let mut order = (0..scores.len()).collect::<Vec<_>>();
        order.sort_by(|&left, &right| scores[left].total_cmp(&scores[right]));

        // (min_score, max_score, positives, count)
        let mut stack: Vec<(f64, f64, f64, f64)> = Vec::new();
        let mut position = 0;
        while position < order.len() {
            let score = scores[order[position]];
            let (mut positives, mut count) = (0.0, 0.0);
            while position < order.len() && scores[order[position]] == score {
                positives += f64::from(u8::from(labels[order[position]]));
                count += 1.0;
                position += 1;
            }
            stack.push((score, score, positives, count));
            while stack.len() >= 2 {
                let last = stack[stack.len() - 1];
                let previous = stack[stack.len() - 2];
                if previous.2 / previous.3 < last.2 / last.3 {
                    break;
                }
                stack.pop();
                let merged = stack.last_mut().expect("stack has two blocks");
                merged.1 = last.1;
                merged.2 += last.2;
                merged.3 += last.3;
            }
        }
        Ok(Self {
            blocks: stack
                .into_iter()
                .map(|(min_score, max_score, positives, count)| IsotonicBlock {
                    min_score,
                    max_score,
                    value: positives / count,
                })
                .collect(),
        })
    }

    /// Value of the last block whose `min_score <= score`; scores below the
    /// first block take the first block's value.
    pub fn apply(&self, score: f64) -> f64 {
        let after = self
            .blocks
            .partition_point(|block| block.min_score <= score);
        self.blocks[after.saturating_sub(1)].value
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::metrics::average_precision;

    #[test]
    fn brier_score_of_perfect_and_constant_predictions() {
        let labels = [true, false, true, false];
        assert_abs_diff_eq!(brier_score(&[1.0, 0.0, 1.0, 0.0], &labels).unwrap(), 0.0);
        assert_abs_diff_eq!(brier_score(&[0.5; 4], &labels).unwrap(), 0.25);
    }

    #[test]
    fn ece_is_zero_when_every_bin_matches_its_observed_rate() {
        // Bin [0.2, 0.3): mean 0.25, 1 positive out of 4. Bin [0.7, 0.8):
        // mean 0.75, 3 positives out of 4.
        let probabilities = [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75];
        let labels = [true, false, false, false, true, true, true, false];
        assert_abs_diff_eq!(
            expected_calibration_error(&probabilities, &labels, 10).unwrap(),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn ece_is_the_count_weighted_gap() {
        // Bin 1: 2 rows, mean 0.1, rate 0.5 -> gap 0.4. Bin 9: 2 rows, mean
        // 0.95 (0.9 and 1.0 share the last bin), rate 1.0 -> gap 0.05.
        let probabilities = [0.1, 0.1, 0.9, 1.0];
        let labels = [true, false, true, true];
        let expected = 0.5 * 0.4 + 0.5 * 0.05;
        assert_abs_diff_eq!(
            expected_calibration_error(&probabilities, &labels, 10).unwrap(),
            expected,
            epsilon = 1e-12
        );
        let bins = reliability_bins(&probabilities, &labels, 10).unwrap();
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[9].count, 2);
        assert_eq!(bins[0].count, 0);
        assert_eq!(bins[0].mean_predicted, None);
    }

    #[test]
    fn platt_scaling_repairs_overconfident_scores_without_changing_ranking() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut scores = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..4000 {
            let true_logit: f64 = rng.gen_range(-3.0..3.0);
            let probability = 1.0 / (1.0 + (-true_logit).exp());
            labels.push(rng.gen::<f64>() < probability);
            // Overconfident: the score's logit is three times the truth.
            scores.push(1.0 / (1.0 + (-3.0 * true_logit).exp()));
        }
        let platt = PlattScaling::fit(&scores, &labels).unwrap();
        assert!(platt.a > 0.0, "a = {}", platt.a);
        assert!((platt.a - 1.0 / 3.0).abs() < 0.1, "a = {}", platt.a);
        let calibrated = scores.iter().map(|&s| platt.apply(s)).collect::<Vec<_>>();
        assert!(
            brier_score(&calibrated, &labels).unwrap() < brier_score(&scores, &labels).unwrap()
        );
        assert!(
            expected_calibration_error(&calibrated, &labels, 10).unwrap()
                < expected_calibration_error(&scores, &labels, 10).unwrap()
        );
        assert_abs_diff_eq!(
            average_precision(&calibrated, &labels).unwrap(),
            average_precision(&scores, &labels).unwrap(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn isotonic_keeps_already_monotone_labels() {
        let scores = [0.1, 0.2, 0.3, 0.4];
        let labels = [false, false, true, true];
        let isotonic = IsotonicRegression::fit(&scores, &labels).unwrap();
        let fitted = scores
            .iter()
            .map(|&s| isotonic.apply(s))
            .collect::<Vec<_>>();
        assert_eq!(fitted, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn isotonic_pools_violators_and_is_non_decreasing() {
        let scores = [0.1, 0.2, 0.3, 0.4];
        let labels = [false, true, false, true];
        let isotonic = IsotonicRegression::fit(&scores, &labels).unwrap();
        assert_abs_diff_eq!(isotonic.apply(0.2), 0.5);
        assert_abs_diff_eq!(isotonic.apply(0.3), 0.5);
        assert_abs_diff_eq!(isotonic.apply(0.1), 0.0);
        assert_abs_diff_eq!(isotonic.apply(0.4), 1.0);
        // Below the first block and between blocks.
        assert_abs_diff_eq!(isotonic.apply(0.0), 0.0);
        assert_abs_diff_eq!(isotonic.apply(0.35), 0.5);

        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let scores = (0..500).map(|_| rng.gen::<f64>()).collect::<Vec<_>>();
        let labels = scores
            .iter()
            .map(|&s| rng.gen::<f64>() < s)
            .collect::<Vec<_>>();
        let isotonic = IsotonicRegression::fit(&scores, &labels).unwrap();
        let mut previous = f64::NEG_INFINITY;
        for step in 0..=1000 {
            let value = isotonic.apply(step as f64 / 1000.0);
            assert!(value >= previous);
            previous = value;
        }
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(
            brier_score(&[0.5], &[true, false]),
            Err(CalibrationError::InvalidLengths)
        );
        assert_eq!(
            brier_score(&[1.5], &[true]),
            Err(CalibrationError::InvalidScore)
        );
        assert_eq!(
            reliability_bins(&[0.5], &[true], 0),
            Err(CalibrationError::InvalidBins)
        );
        assert_eq!(
            PlattScaling::fit(&[0.2, 0.3], &[true, true]),
            Err(CalibrationError::SingleClass)
        );
    }
}
