use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MatrixError {
    #[error("matrix dimensions {rows}x{cols} require {expected} values, got {actual}")]
    InvalidShape {
        rows: usize,
        cols: usize,
        expected: usize,
        actual: usize,
    },
    #[error("row {row} is out of bounds for a matrix with {rows} rows")]
    RowOutOfBounds { row: usize, rows: usize },
    #[error("all rows must have the same length")]
    RaggedRows,
    #[error("a matrix must contain at least one row and one column")]
    Empty,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl DenseMatrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, MatrixError> {
        let expected = rows.saturating_mul(cols);
        if rows == 0 || cols == 0 {
            return Err(MatrixError::Empty);
        }
        if data.len() != expected {
            return Err(MatrixError::InvalidShape {
                rows,
                cols,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    pub fn from_rows(rows: Vec<Vec<f64>>) -> Result<Self, MatrixError> {
        let Some(first) = rows.first() else {
            return Err(MatrixError::Empty);
        };
        if first.is_empty() {
            return Err(MatrixError::Empty);
        }
        let cols = first.len();
        if rows.iter().any(|row| row.len() != cols) {
            return Err(MatrixError::RaggedRows);
        }
        let row_count = rows.len();
        let data = rows.into_iter().flatten().collect();
        Self::new(row_count, cols, data)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, row: usize) -> Result<&[f64], MatrixError> {
        if row >= self.rows {
            return Err(MatrixError::RowOutOfBounds {
                row,
                rows: self.rows,
            });
        }
        Ok(self.row_unchecked(row))
    }

    pub(crate) fn row_unchecked(&self, row: usize) -> &[f64] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_contiguous_views() {
        let matrix = DenseMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(matrix.row(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(matrix.row(1).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn rejects_invalid_shapes_and_ragged_rows() {
        assert!(matches!(
            DenseMatrix::new(2, 2, vec![1.0]),
            Err(MatrixError::InvalidShape { .. })
        ));
        assert_eq!(
            DenseMatrix::from_rows(vec![vec![1.0], vec![2.0, 3.0]]),
            Err(MatrixError::RaggedRows)
        );
    }
}
