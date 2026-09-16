#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace tp3 {

// Dense row-major matrix of doubles. Every shape mismatch throws
// std::invalid_argument naming both shapes. Plain loops, no external code.
class Matrix {
public:
    Matrix();
    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0);

    // Builds a matrix from a list of rows; throws on ragged input.
    static Matrix from_rows(const std::vector<std::vector<double>>& rows);

    // Fills in row-major order from one uniform_real_distribution(lo, hi) draw per element.
    static Matrix random(std::size_t rows, std::size_t cols, std::mt19937_64& rng, double lo, double hi);

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    double& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
    double operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

    const std::vector<double>& data() const { return data_; }

    // Row i as a 1 x cols matrix.
    Matrix row(std::size_t i) const;

    Matrix operator+(const Matrix& rhs) const;
    Matrix operator-(const Matrix& rhs) const;
    Matrix operator*(const Matrix& rhs) const;  // matrix product
    Matrix operator*(double scalar) const;
    Matrix operator+(double scalar) const;

    Matrix transpose() const;
    Matrix hadamard(const Matrix& rhs) const;  // element-wise product
    Matrix apply(double (*fn)(double)) const;

    bool operator==(const Matrix& rhs) const;
    bool operator!=(const Matrix& rhs) const { return !(*this == rhs); }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

}  // namespace tp3
