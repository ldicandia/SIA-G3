// VAL-08 groundwork for Phase 2: the Matrix operations backprop will run on,
// and the shape-mismatch exceptions that name both shapes.
#include <random>
#include <stdexcept>

#include "doctest.h"
#include "matrix.hpp"

using tp3::Matrix;

namespace {
double square(double v) { return v * v; }
}  // namespace

TEST_CASE("ctor fills, rows/cols report the shape") {
    const Matrix m(2, 3, 1.5);
    CHECK(m.rows() == 2);
    CHECK(m.cols() == 3);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            CHECK(m(i, j) == 1.5);
        }
    }
    const Matrix zero(2, 2);
    CHECK(zero(1, 1) == 0.0);
    const Matrix empty;
    CHECK(empty.rows() == 0);
    CHECK(empty.cols() == 0);
}

TEST_CASE("from_rows builds row-major and rejects ragged input") {
    const Matrix m = Matrix::from_rows({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    CHECK(m.rows() == 3);
    CHECK(m.cols() == 2);
    CHECK(m(0, 1) == 2.0);
    CHECK(m(2, 0) == 5.0);
    CHECK_THROWS_AS(Matrix::from_rows({{1.0, 2.0}, {3.0}}), std::invalid_argument);
}

TEST_CASE("operator() reads and writes") {
    Matrix m(2, 2);
    m(0, 1) = 7.0;
    m(1, 0) = -3.0;
    CHECK(m(0, 1) == 7.0);
    CHECK(m(1, 0) == -3.0);
    CHECK(m(0, 0) == 0.0);
}

TEST_CASE("+ and - are elementwise") {
    const Matrix a = Matrix::from_rows({{1.0, 2.0}, {3.0, 4.0}});
    const Matrix b = Matrix::from_rows({{10.0, 20.0}, {30.0, 40.0}});
    CHECK(a + b == Matrix::from_rows({{11.0, 22.0}, {33.0, 44.0}}));
    CHECK(b - a == Matrix::from_rows({{9.0, 18.0}, {27.0, 36.0}}));
}

TEST_CASE("matrix product 2x3 * 3x2") {
    const Matrix a = Matrix::from_rows({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    const Matrix b = Matrix::from_rows({{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}});
    CHECK(a * b == Matrix::from_rows({{58.0, 64.0}, {139.0, 154.0}}));
}

TEST_CASE("scalar multiply and add") {
    const Matrix a = Matrix::from_rows({{1.0, -2.0}, {0.5, 4.0}});
    CHECK(a * 2.0 == Matrix::from_rows({{2.0, -4.0}, {1.0, 8.0}}));
    CHECK(a + 1.0 == Matrix::from_rows({{2.0, -1.0}, {1.5, 5.0}}));
}

TEST_CASE("transpose of 2x3 is 3x2 with swapped entries") {
    const Matrix a = Matrix::from_rows({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    const Matrix t = a.transpose();
    CHECK(t.rows() == 3);
    CHECK(t.cols() == 2);
    CHECK(t == Matrix::from_rows({{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}));
    CHECK(t.transpose() == a);
}

TEST_CASE("hadamard is the elementwise product") {
    const Matrix a = Matrix::from_rows({{1.0, 2.0}, {3.0, 4.0}});
    const Matrix b = Matrix::from_rows({{2.0, 0.5}, {-1.0, 2.0}});
    CHECK(a.hadamard(b) == Matrix::from_rows({{2.0, 1.0}, {-3.0, 8.0}}));
}

TEST_CASE("apply maps a plain function over every element") {
    const Matrix a = Matrix::from_rows({{1.0, -2.0}, {3.0, 0.0}});
    CHECK(a.apply(square) == Matrix::from_rows({{1.0, 4.0}, {9.0, 0.0}}));
}

TEST_CASE("row(i) is a 1 x cols matrix; out-of-range throws") {
    const Matrix a = Matrix::from_rows({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    const Matrix r = a.row(1);
    CHECK(r.rows() == 1);
    CHECK(r.cols() == 3);
    CHECK(r == Matrix::from_rows({{4.0, 5.0, 6.0}}));
    CHECK_THROWS_AS(a.row(2), std::invalid_argument);
}

TEST_CASE("random fills within [lo, hi] and is reproducible from the seed") {
    std::mt19937_64 rng_a(42);
    std::mt19937_64 rng_b(42);
    const Matrix a = Matrix::random(2, 3, rng_a, -0.5, 0.5);
    const Matrix b = Matrix::random(2, 3, rng_b, -0.5, 0.5);
    CHECK(a.rows() == 2);
    CHECK(a.cols() == 3);
    for (double v : a.data()) {
        CHECK(v >= -0.5);
        CHECK(v <= 0.5);
    }
    CHECK(a == b);
    const Matrix c = Matrix::random(2, 3, rng_a, -0.5, 0.5);  // further draws differ
    CHECK(c != a);
}

TEST_CASE("shape mismatches throw std::invalid_argument") {
    const Matrix a(2, 3);
    const Matrix b(3, 2);
    const Matrix c(2, 2);
    CHECK_THROWS_AS(a + b, std::invalid_argument);
    CHECK_THROWS_AS(a - b, std::invalid_argument);
    CHECK_THROWS_AS(a * c, std::invalid_argument);
    CHECK_THROWS_AS(a.hadamard(b), std::invalid_argument);
    CHECK_NOTHROW(a * b);
}
