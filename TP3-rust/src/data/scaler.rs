use thiserror::Error;

use crate::matrix::{DenseMatrix, MatrixError};

#[derive(Clone, Debug)]
pub struct StandardScaler {
    means: Vec<f64>,
    scales: Vec<f64>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ScalerError {
    #[error("cannot fit a scaler on an empty selection")]
    EmptySelection,
    #[error("row index {index} is out of bounds")]
    InvalidIndex { index: usize },
    #[error("input has {actual} columns, scaler expects {expected}")]
    Dimension { expected: usize, actual: usize },
    #[error(transparent)]
    Matrix(#[from] MatrixError),
}

impl StandardScaler {
    pub fn fit(matrix: &DenseMatrix, indices: &[usize]) -> Result<Self, ScalerError> {
        if indices.is_empty() {
            return Err(ScalerError::EmptySelection);
        }
        let mut means = vec![0.0; matrix.cols()];
        for &index in indices {
            if index >= matrix.rows() {
                return Err(ScalerError::InvalidIndex { index });
            }
            for (mean, value) in means.iter_mut().zip(matrix.row_unchecked(index)) {
                *mean += value;
            }
        }
        for mean in &mut means {
            *mean /= indices.len() as f64;
        }

        let mut scales = vec![0.0; matrix.cols()];
        for &index in indices {
            for ((scale, value), mean) in scales
                .iter_mut()
                .zip(matrix.row_unchecked(index))
                .zip(&means)
            {
                *scale += (value - mean).powi(2);
            }
        }
        for scale in &mut scales {
            *scale = (*scale / indices.len() as f64).sqrt();
            if *scale <= f64::EPSILON {
                *scale = 1.0;
            }
        }
        Ok(Self { means, scales })
    }

    pub fn transform(&self, matrix: &DenseMatrix) -> Result<DenseMatrix, ScalerError> {
        if matrix.cols() != self.means.len() {
            return Err(ScalerError::Dimension {
                expected: self.means.len(),
                actual: matrix.cols(),
            });
        }
        let data = matrix
            .as_slice()
            .chunks_exact(matrix.cols())
            .flat_map(|row| {
                row.iter()
                    .zip(&self.means)
                    .zip(&self.scales)
                    .map(|((&value, &mean), &scale)| (value - mean) / scale)
            })
            .collect();
        Ok(DenseMatrix::new(matrix.rows(), matrix.cols(), data)?)
    }

    pub fn means(&self) -> &[f64] {
        &self.means
    }

    pub fn scales(&self) -> &[f64] {
        &self.scales
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn scaler_fits_only_selected_rows() {
        let matrix =
            DenseMatrix::from_rows(vec![vec![1.0, 4.0], vec![3.0, 4.0], vec![100.0, 4.0]]).unwrap();
        let scaler = StandardScaler::fit(&matrix, &[0, 1]).unwrap();
        assert_abs_diff_eq!(scaler.means()[0], 2.0);
        assert_abs_diff_eq!(scaler.scales()[0], 1.0);
        assert_abs_diff_eq!(scaler.scales()[1], 1.0);

        let transformed = scaler.transform(&matrix).unwrap();
        assert_eq!(transformed.row(0).unwrap(), &[-1.0, 0.0]);
        assert_eq!(transformed.row(1).unwrap(), &[1.0, 0.0]);
        assert_eq!(transformed.row(2).unwrap(), &[98.0, 0.0]);
    }
}
