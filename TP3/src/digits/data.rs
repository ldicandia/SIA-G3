use std::path::Path;

use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

use crate::matrix::{DenseMatrix, MatrixError};

pub const DIGIT_CLASSES: usize = 10;
pub const IMAGE_PIXELS: usize = 784;

#[derive(Clone, Debug)]
pub struct DigitDataset {
    pub features: DenseMatrix,
    pub labels: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct DigitSplit {
    pub train: Vec<usize>,
    pub validation: Vec<usize>,
}

#[derive(Debug, Error)]
pub enum DigitDataError {
    #[error("could not read CSV: {0}")]
    Csv(#[from] csv::Error),
    #[error("dataset must contain label and image columns")]
    MissingColumns,
    #[error("row {row} has an invalid digit label")]
    InvalidLabel { row: usize },
    #[error("row {row} has an invalid image value")]
    InvalidImage { row: usize },
    #[error("row {row} has {actual} pixels, expected {expected}")]
    InvalidPixelCount {
        row: usize,
        expected: usize,
        actual: usize,
    },
    #[error("digit dataset cannot be empty")]
    Empty,
    #[error(transparent)]
    Matrix(#[from] MatrixError),
}

pub fn load_digit_dataset(path: &Path) -> Result<DigitDataset, DigitDataError> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let label_column = headers
        .iter()
        .position(|header| header == "label")
        .ok_or(DigitDataError::MissingColumns)?;
    let image_column = headers
        .iter()
        .position(|header| header == "image")
        .ok_or(DigitDataError::MissingColumns)?;

    let mut labels = Vec::new();
    let mut values = Vec::new();
    for (index, record) in reader.records().enumerate() {
        let row = index + 2;
        let record = record?;
        let label = record
            .get(label_column)
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value < DIGIT_CLASSES)
            .ok_or(DigitDataError::InvalidLabel { row })?;
        let image = record
            .get(image_column)
            .ok_or(DigitDataError::InvalidImage { row })?;
        let pixels = parse_image(image).ok_or(DigitDataError::InvalidImage { row })?;
        if pixels.len() != IMAGE_PIXELS {
            return Err(DigitDataError::InvalidPixelCount {
                row,
                expected: IMAGE_PIXELS,
                actual: pixels.len(),
            });
        }
        labels.push(label);
        values.extend(pixels);
    }
    if labels.is_empty() {
        return Err(DigitDataError::Empty);
    }
    Ok(DigitDataset {
        features: DenseMatrix::new(labels.len(), IMAGE_PIXELS, values)?,
        labels,
    })
}

fn parse_image(value: &str) -> Option<Vec<f64>> {
    let contents = value.trim().strip_prefix('[')?.strip_suffix(']')?;
    contents
        .split(',')
        .map(|pixel| {
            pixel
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
        })
        .collect()
}

pub fn stratified_digit_split(labels: &[usize], validation_ratio: f64, seed: u64) -> DigitSplit {
    let mut groups = vec![Vec::new(); DIGIT_CLASSES];
    for (index, &label) in labels.iter().enumerate() {
        groups[label].push(index);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut train = Vec::new();
    let mut validation = Vec::new();
    for group in &mut groups {
        group.shuffle(&mut rng);
        let validation_count = (group.len() as f64 * validation_ratio).round() as usize;
        validation.extend_from_slice(&group[..validation_count]);
        train.extend_from_slice(&group[validation_count..]);
    }
    train.shuffle(&mut rng);
    validation.shuffle(&mut rng);
    DigitSplit { train, validation }
}

pub fn class_counts(labels: &[usize], indices: &[usize]) -> [usize; DIGIT_CLASSES] {
    let mut counts = [0; DIGIT_CLASSES];
    for &index in indices {
        counts[labels[index]] += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn split_is_stratified_disjoint_and_complete() {
        let labels = (0..100).map(|index| index % 10).collect::<Vec<_>>();
        let split = stratified_digit_split(&labels, 0.2, 42);
        assert_eq!(split.train.len(), 80);
        assert_eq!(split.validation.len(), 20);
        assert_eq!(class_counts(&labels, &split.validation), [2; 10]);
        let all = split
            .train
            .iter()
            .chain(&split.validation)
            .copied()
            .collect::<HashSet<_>>();
        assert_eq!(all.len(), labels.len());
    }
}
