#include "matrix.hpp"

#include <stdexcept>
#include <string>

namespace tp3 {

namespace {

std::string shape(const Matrix& m) {
    return std::to_string(m.rows()) + "x" + std::to_string(m.cols());
}

void require_same_shape(const char* op, const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument(std::string(op) + ": " + shape(a) + " vs " + shape(b));
    }
}

}  // namespace

Matrix::Matrix() : rows_(0), cols_(0), data_() {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double fill)
    : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

Matrix Matrix::from_rows(const std::vector<std::vector<double>>& rows) {
    if (rows.empty()) {
        return Matrix();
    }
    const std::size_t cols = rows.front().size();
    Matrix m(rows.size(), cols);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].size() != cols) {
            throw std::invalid_argument("from_rows: ragged row " + std::to_string(i) + " has " +
                                        std::to_string(rows[i].size()) + " columns, expected " +
                                        std::to_string(cols));
        }
        for (std::size_t j = 0; j < cols; ++j) {
            m(i, j) = rows[i][j];
        }
    }
    return m;
}

Matrix Matrix::random(std::size_t rows, std::size_t cols, std::mt19937_64& rng, double lo, double hi) {
    Matrix m(rows, cols);
    std::uniform_real_distribution<double> dist(lo, hi);
    for (std::size_t k = 0; k < m.data_.size(); ++k) {
        m.data_[k] = dist(rng);
    }
    return m;
}

Matrix Matrix::row(std::size_t i) const {
    if (i >= rows_) {
        throw std::invalid_argument("row: index " + std::to_string(i) + " out of range for " + shape(*this));
    }
    Matrix r(1, cols_);
    for (std::size_t j = 0; j < cols_; ++j) {
        r(0, j) = (*this)(i, j);
    }
    return r;
}

Matrix Matrix::operator+(const Matrix& rhs) const {
    require_same_shape("matrix add", *this, rhs);
    Matrix out(rows_, cols_);
    for (std::size_t k = 0; k < data_.size(); ++k) {
        out.data_[k] = data_[k] + rhs.data_[k];
    }
    return out;
}

Matrix Matrix::operator-(const Matrix& rhs) const {
    require_same_shape("matrix subtract", *this, rhs);
    Matrix out(rows_, cols_);
    for (std::size_t k = 0; k < data_.size(); ++k) {
        out.data_[k] = data_[k] - rhs.data_[k];
    }
    return out;
}

Matrix Matrix::operator*(const Matrix& rhs) const {
    if (cols_ != rhs.rows_) {
        throw std::invalid_argument("matrix product: " + shape(*this) + " * " + shape(rhs));
    }
    Matrix out(rows_, rhs.cols_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t k = 0; k < cols_; ++k) {
            const double a = (*this)(i, k);
            for (std::size_t j = 0; j < rhs.cols_; ++j) {
                out(i, j) += a * rhs(k, j);
            }
        }
    }
    return out;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix out(rows_, cols_);
    for (std::size_t k = 0; k < data_.size(); ++k) {
        out.data_[k] = data_[k] * scalar;
    }
    return out;
}

Matrix Matrix::operator+(double scalar) const {
    Matrix out(rows_, cols_);
    for (std::size_t k = 0; k < data_.size(); ++k) {
        out.data_[k] = data_[k] + scalar;
    }
    return out;
}

Matrix Matrix::transpose() const {
    Matrix out(cols_, rows_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            out(j, i) = (*this)(i, j);
        }
    }
    return out;
}

Matrix Matrix::hadamard(const Matrix& rhs) const {
    require_same_shape("hadamard", *this, rhs);
    Matrix out(rows_, cols_);
    for (std::size_t k = 0; k < data_.size(); ++k) {
        out.data_[k] = data_[k] * rhs.data_[k];
    }
    return out;
}

Matrix Matrix::apply(double (*fn)(double)) const {
    Matrix out(rows_, cols_);
    for (std::size_t k = 0; k < data_.size(); ++k) {
        out.data_[k] = fn(data_[k]);
    }
    return out;
}

bool Matrix::operator==(const Matrix& rhs) const {
    return rows_ == rhs.rows_ && cols_ == rhs.cols_ && data_ == rhs.data_;
}

}  // namespace tp3
