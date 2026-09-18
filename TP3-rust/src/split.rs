use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct DatasetSplit {
    pub train: Vec<usize>,
    pub validation: Vec<usize>,
    pub test: Vec<usize>,
}

#[derive(Debug, Error, PartialEq)]
pub enum SplitError {
    #[error("targets cannot be empty")]
    Empty,
    #[error("ratios must be positive and sum to less than one")]
    InvalidRatios,
    #[error("at least two stratification bins are required")]
    InvalidBins,
    #[error("all stratification targets must be finite and in [0, 1]")]
    InvalidTarget,
}

pub fn stratified_split(
    targets: &[f64],
    train_ratio: f64,
    validation_ratio: f64,
    bins: usize,
    seed: u64,
) -> Result<DatasetSplit, SplitError> {
    if targets.is_empty() {
        return Err(SplitError::Empty);
    }
    if !(train_ratio > 0.0 && validation_ratio > 0.0 && train_ratio + validation_ratio < 1.0) {
        return Err(SplitError::InvalidRatios);
    }
    if bins < 2 {
        return Err(SplitError::InvalidBins);
    }
    if targets
        .iter()
        .any(|target| !target.is_finite() || !(0.0..=1.0).contains(target))
    {
        return Err(SplitError::InvalidTarget);
    }

    let mut strata = vec![Vec::new(); bins];
    for (index, &target) in targets.iter().enumerate() {
        let bin = ((target * bins as f64).floor() as usize).min(bins - 1);
        strata[bin].push(index);
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut split = DatasetSplit {
        train: Vec::new(),
        validation: Vec::new(),
        test: Vec::new(),
    };
    for stratum in &mut strata {
        stratum.shuffle(&mut rng);
        let train_end = (stratum.len() as f64 * train_ratio).round() as usize;
        let validation_count = (stratum.len() as f64 * validation_ratio).round() as usize;
        let validation_end = (train_end + validation_count).min(stratum.len());
        split.train.extend_from_slice(&stratum[..train_end]);
        split
            .validation
            .extend_from_slice(&stratum[train_end..validation_end]);
        split.test.extend_from_slice(&stratum[validation_end..]);
    }
    split.train.shuffle(&mut rng);
    split.validation.shuffle(&mut rng);
    split.test.shuffle(&mut rng);
    Ok(split)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn split_is_deterministic_disjoint_and_complete() {
        let targets = (0..100)
            .map(|index| index as f64 / 99.0)
            .collect::<Vec<_>>();
        let first = stratified_split(&targets, 0.7, 0.15, 10, 42).unwrap();
        let second = stratified_split(&targets, 0.7, 0.15, 10, 42).unwrap();
        assert_eq!(first.train, second.train);
        assert_eq!(first.validation, second.validation);
        assert_eq!(first.test, second.test);

        let all = first
            .train
            .iter()
            .chain(&first.validation)
            .chain(&first.test)
            .copied()
            .collect::<HashSet<_>>();
        assert_eq!(all.len(), targets.len());
        assert_eq!(first.train.len(), 70);
        assert_eq!(first.validation.len(), 20);
        assert_eq!(first.test.len(), 10);
    }
}
