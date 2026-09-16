#include "validation/datasets.hpp"

#include <cmath>

namespace tp3 {

namespace {

// n x 1 column of draws from U(lo, hi), in index order, so the seed reproduces the samples.
Matrix sample_uniform_column(std::size_t n, std::mt19937_64& rng, double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    Matrix x(n, 1, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 0) = dist(rng);
    }
    return x;
}

}  // namespace

Dataset and_dataset() {
    Dataset d;
    d.X = Matrix::from_rows({{-1.0, 1.0}, {1.0, -1.0}, {-1.0, -1.0}, {1.0, 1.0}});
    d.y = Matrix::from_rows({{-1.0}, {-1.0}, {-1.0}, {1.0}});
    return d;
}

Dataset linear_dataset(std::size_t n, std::mt19937_64& rng) {
    Dataset d;
    d.X = sample_uniform_column(n, rng, -2.0, 2.0);
    d.y = d.X;
    return d;
}

Dataset tanh_dataset(std::size_t n, std::mt19937_64& rng) {
    Dataset d;
    d.X = sample_uniform_column(n, rng, -2.0, 2.0);
    d.y = d.X.apply([](double x) { return std::tanh(x); });
    return d;
}

Dataset xor_dataset() {
    Dataset d;
    d.X = Matrix::from_rows({{-1.0, 1.0}, {1.0, -1.0}, {-1.0, -1.0}, {1.0, 1.0}});
    d.y = Matrix::from_rows({{1.0}, {1.0}, {-1.0}, {-1.0}});
    return d;
}

}  // namespace tp3
