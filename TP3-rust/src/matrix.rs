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

// Row-major batch kernels used by the mini-batch MLP trainer. A batch of `m`
// samples is an `m x k` matrix, and a dense layer stores its weights as an
// `n x k` matrix (one row per output neuron), so:
//
//   forward:            Z = A Wᵀ + b          (`affine_nt`)
//   backward to inputs: D_prev = D W          (`matmul_nn`)
//   weight gradient:    G = Dᵀ A              (`matmul_tn`)
//
// The kernels are sequential: the trainer parallelizes over sub-batches,
// which keeps every thread busy with coarse work instead of waking the pool
// for each small product. Inner loops are contiguous dot products and axpy
// updates, which the compiler vectorizes.

/// `out[m x n] = a[m x k] · w[n x k]ᵀ + bias[n]`.
pub fn affine_nt(a: &[f64], k: usize, w: &[f64], bias: &[f64], out: &mut [f64]) {
    let n = bias.len();
    debug_assert_eq!(w.len(), n * k);
    debug_assert_eq!(a.len() / k, out.len() / n);
    for (out_row, a_row) in out.chunks_exact_mut(n).zip(a.chunks_exact(k)) {
        for (j, value) in out_row.iter_mut().enumerate() {
            *value = fast_dot(a_row, &w[j * k..(j + 1) * k]) + bias[j];
        }
    }
}

/// `out[m x k] = d[m x n] · w[n x k]`.
pub fn matmul_nn(d: &[f64], n: usize, w: &[f64], out: &mut [f64]) {
    let k = w.len() / n;
    debug_assert_eq!(d.len() / n, out.len() / k);
    for (out_row, d_row) in out.chunks_exact_mut(k).zip(d.chunks_exact(n)) {
        out_row.fill(0.0);
        for (j, &scale) in d_row.iter().enumerate() {
            if scale != 0.0 {
                axpy(scale, &w[j * k..(j + 1) * k], out_row);
            }
        }
    }
}

/// `g[n x k] = d[m x n]ᵀ · a[m x k]` and `bias_g[n] = column sums of d`.
pub fn matmul_tn(d: &[f64], n: usize, a: &[f64], k: usize, g: &mut [f64], bias_g: &mut [f64]) {
    let m = d.len() / n;
    debug_assert_eq!(a.len(), m * k);
    g.fill(0.0);
    bias_g.fill(0.0);
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let scale = d[i * n + j];
            if scale != 0.0 {
                axpy(scale, a_row, &mut g[j * k..(j + 1) * k]);
                bias_g[j] += scale;
            }
        }
    }
}

/// Dot product with independent partial sums so it can be vectorized.
#[inline]
pub fn fast_dot(left: &[f64], right: &[f64]) -> f64 {
    let mut partial = [0.0; 8];
    let (left_chunks, left_tail) = left.as_chunks::<8>();
    let (right_chunks, right_tail) = right.as_chunks::<8>();
    let tail = left_tail
        .iter()
        .zip(right_tail)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    for (a, b) in left_chunks.iter().zip(right_chunks) {
        for ((sum, x), y) in partial.iter_mut().zip(a).zip(b) {
            *sum += x * y;
        }
    }
    partial.iter().sum::<f64>() + tail
}

/// `y += scale * x`.
#[inline]
pub fn axpy(scale: f64, x: &[f64], y: &mut [f64]) {
    for (target, &value) in y.iter_mut().zip(x) {
        *target += scale * value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_kernels_match_naive_products() {
        // a: 2x3, w: 2x3 (n=2 outputs), d: 2x2.
        let a = [1.0, 2.0, 3.0, -1.0, 0.5, 2.0];
        let w = [0.1, 0.2, 0.3, -0.4, 0.5, 0.6];
        let bias = [0.01, -0.02];
        let mut z = [0.0; 4];
        affine_nt(&a, 3, &w, &bias, &mut z);
        let expected_z = [
            0.1 + 0.4 + 0.9 + 0.01,
            -0.4 + 1.0 + 1.8 - 0.02,
            -0.1 + 0.1 + 0.6 + 0.01,
            0.4 + 0.25 + 1.2 - 0.02,
        ];
        for (got, want) in z.iter().zip(expected_z) {
            assert!((got - want).abs() < 1e-12);
        }

        let d = [1.0, 2.0, -1.0, 0.5];
        let mut back = [0.0; 6];
        matmul_nn(&d, 2, &w, &mut back);
        let expected_back = [
            0.1 - 0.8,
            0.2 + 1.0,
            0.3 + 1.2,
            -0.1 - 0.2,
            -0.2 + 0.25,
            -0.3 + 0.3,
        ];
        for (got, want) in back.iter().zip(expected_back) {
            assert!((got - want).abs() < 1e-12);
        }

        let mut g = [0.0; 6];
        let mut bias_g = [0.0; 2];
        matmul_tn(&d, 2, &a, 3, &mut g, &mut bias_g);
        let expected_g = [
            1.0 + 1.0,
            2.0 - 0.5,
            3.0 - 2.0,
            2.0 - 0.5,
            4.0 + 0.25,
            6.0 + 1.0,
        ];
        for (got, want) in g.iter().zip(expected_g) {
            assert!((got - want).abs() < 1e-12);
        }
        assert_eq!(bias_g, [0.0, 2.5]);
    }

    #[test]
    fn fast_dot_handles_tails() {
        let left = (0..19).map(f64::from).collect::<Vec<_>>();
        let expected = left.iter().map(|v| v * v).sum::<f64>();
        assert_eq!(fast_dot(&left, &left), expected);
    }

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

    #[test]
    fn rejects_zero_rows_or_columns() {
        assert_eq!(DenseMatrix::new(0, 2, vec![]), Err(MatrixError::Empty));
        assert_eq!(DenseMatrix::new(2, 0, vec![]), Err(MatrixError::Empty));
    }

    #[test]
    fn from_rows_rejects_empty_input() {
        assert_eq!(DenseMatrix::from_rows(vec![]), Err(MatrixError::Empty));
        assert_eq!(
            DenseMatrix::from_rows(vec![vec![]]),
            Err(MatrixError::Empty)
        );
    }

    #[test]
    fn accessors_report_shape() {
        let matrix = DenseMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
    }

    #[test]
    fn row_out_of_bounds_is_reported() {
        let matrix = DenseMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            matrix.row(2),
            Err(MatrixError::RowOutOfBounds { row: 2, rows: 2 })
        );
    }

    #[test]
    fn as_slice_exposes_row_major_data() {
        let matrix = DenseMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(matrix.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn from_rows_matches_equivalent_new_construction() {
        let from_rows = DenseMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let from_new = DenseMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(from_rows, from_new);
    }
}
