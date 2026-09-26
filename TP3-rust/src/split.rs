use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

/// A development set (used for cross-validation and the final refit) and a
/// test set that is consulted exactly once.
#[derive(Clone, Debug)]
pub struct HoldoutSplit {
    pub development: Vec<usize>,
    pub test: Vec<usize>,
}

/// One cross-validation fold: the model trains on `train` and is scored on
/// `validation`. Validation sets of different folds are disjoint.
#[derive(Clone, Debug)]
pub struct Fold {
    pub train: Vec<usize>,
    pub validation: Vec<usize>,
}

#[derive(Debug, Error, PartialEq)]
pub enum SplitError {
    #[error("targets cannot be empty")]
    Empty,
    #[error("test ratio must be in (0, 1)")]
    InvalidRatio,
    #[error("at least two stratification bins are required")]
    InvalidBins,
    #[error("at least two folds are required and each fold needs rows")]
    InvalidFolds,
    #[error("all stratification targets must be finite and in [0, 1]")]
    InvalidTarget,
}

/// Stratifies by bands of the [0, 1] target so every split sees the same
/// distribution of BigModel probabilities.
pub fn stratified_holdout(
    targets: &[f64],
    test_ratio: f64,
    bins: usize,
    seed: u64,
) -> Result<HoldoutSplit, SplitError> {
    if !(test_ratio > 0.0 && test_ratio < 1.0) {
        return Err(SplitError::InvalidRatio);
    }
    let all = (0..targets.len()).collect::<Vec<_>>();
    let strata = strata(targets, &all, bins)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut split = HoldoutSplit {
        development: Vec::new(),
        test: Vec::new(),
    };
    for mut stratum in strata {
        stratum.shuffle(&mut rng);
        let test_count = (stratum.len() as f64 * test_ratio).round() as usize;
        split.test.extend_from_slice(&stratum[..test_count]);
        split.development.extend_from_slice(&stratum[test_count..]);
    }
    split.development.shuffle(&mut rng);
    split.test.shuffle(&mut rng);
    Ok(split)
}

/// Stratified k-fold over `indices`: each stratum is shuffled and dealt
/// round-robin into the folds, so every fold keeps the target distribution.
pub fn stratified_folds(
    targets: &[f64],
    indices: &[usize],
    folds: usize,
    bins: usize,
    seed: u64,
) -> Result<Vec<Fold>, SplitError> {
    if folds < 2 || indices.len() < folds {
        return Err(SplitError::InvalidFolds);
    }
    let strata = strata(targets, indices, bins)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut assignments = vec![Vec::new(); folds];
    let mut next = 0;
    for mut stratum in strata {
        stratum.shuffle(&mut rng);
        for index in stratum {
            assignments[next % folds].push(index);
            next += 1;
        }
    }
    Ok((0..folds)
        .map(|fold| Fold {
            train: assignments
                .iter()
                .enumerate()
                .filter(|&(other, _)| other != fold)
                .flat_map(|(_, rows)| rows.iter().copied())
                .collect(),
            validation: assignments[fold].clone(),
        })
        .collect())
}

fn strata(targets: &[f64], indices: &[usize], bins: usize) -> Result<Vec<Vec<usize>>, SplitError> {
    if targets.is_empty() || indices.is_empty() {
        return Err(SplitError::Empty);
    }
    if bins < 2 {
        return Err(SplitError::InvalidBins);
    }
    let mut strata = vec![Vec::new(); bins];
    for &index in indices {
        let target = *targets.get(index).ok_or(SplitError::InvalidTarget)?;
        if !target.is_finite() || !(0.0..=1.0).contains(&target) {
            return Err(SplitError::InvalidTarget);
        }
        let bin = ((target * bins as f64).floor() as usize).min(bins - 1);
        strata[bin].push(index);
    }
    Ok(strata)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn targets() -> Vec<f64> {
        (0..100).map(|index| index as f64 / 99.0).collect()
    }

    #[test]
    fn holdout_is_deterministic_disjoint_and_complete() {
        let targets = targets();
        let first = stratified_holdout(&targets, 0.2, 10, 42).unwrap();
        let second = stratified_holdout(&targets, 0.2, 10, 42).unwrap();
        assert_eq!(first.development, second.development);
        assert_eq!(first.test, second.test);
        assert_eq!(first.test.len(), 20);
        let all = first
            .development
            .iter()
            .chain(&first.test)
            .copied()
            .collect::<HashSet<_>>();
        assert_eq!(all.len(), targets.len());
    }

    #[test]
    fn folds_partition_the_development_rows() {
        let targets = targets();
        let development = (0..80).collect::<Vec<_>>();
        let folds = stratified_folds(&targets, &development, 5, 10, 7).unwrap();
        assert_eq!(folds.len(), 5);
        let mut seen = HashSet::new();
        for fold in &folds {
            assert_eq!(fold.validation.len(), 16);
            assert_eq!(fold.train.len() + fold.validation.len(), 80);
            let train = fold.train.iter().collect::<HashSet<_>>();
            assert!(fold.validation.iter().all(|row| !train.contains(row)));
            for &row in &fold.validation {
                assert!(seen.insert(row));
            }
        }
        assert_eq!(seen.len(), 80);
    }
}
