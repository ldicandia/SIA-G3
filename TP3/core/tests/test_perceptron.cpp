// ENG-02 / VAL-08: the simple perceptron's online update on known inputs
// (K-06: delta = (y - o) * df(h), w += lr*delta*x, b += lr*delta), argument
// validation and the loss-history length.
#include <random>
#include <stdexcept>

#include "doctest.h"
#include "perceptron.hpp"
#include "validation/datasets.hpp"

using tp3::Matrix;
using tp3::SimplePerceptron;

TEST_CASE("one online step with known weights") {
    std::mt19937_64 rng(1);
    SimplePerceptron p(2, "step", 0.1, rng);
    p.set_weights(Matrix(2, 1, 0.0), 0.0);

    const Matrix X = Matrix::from_rows({{1.0, -1.0}});
    const Matrix y = Matrix::from_rows({{-1.0}});
    const tp3::TrainResult r = p.fit(X, y, 1);

    // h = 0 -> o = +1 -> delta = (-1 - 1) * 1 = -2
    CHECK(p.weights()(0, 0) == doctest::Approx(-0.2));
    CHECK(p.weights()(1, 0) == doctest::Approx(0.2));
    CHECK(p.bias() == doctest::Approx(-0.2));
    CHECK(r.loss_per_epoch.size() == 1);
}

TEST_CASE("identity perceptron one step") {
    std::mt19937_64 rng(1);
    SimplePerceptron p(2, "identity", 0.1, rng);
    p.set_weights(Matrix::from_rows({{0.5}, {0.0}}), 0.0);

    const Matrix X = Matrix::from_rows({{2.0, 0.0}});
    const Matrix y = Matrix::from_rows({{2.0}});
    p.fit(X, y, 1);

    // h = 1 -> o = 1 -> delta = (2 - 1) * 1 = 1
    CHECK(p.weights()(0, 0) == doctest::Approx(0.7));
    CHECK(p.weights()(1, 0) == doctest::Approx(0.0));
    CHECK(p.bias() == doctest::Approx(0.1));
}

TEST_CASE("fit validates its arguments") {
    std::mt19937_64 rng(1);
    SimplePerceptron p(2, "step", 0.1, rng);
    const Matrix X = Matrix::from_rows({{1.0, 1.0}, {-1.0, 1.0}});
    const Matrix y = Matrix::from_rows({{1.0}, {-1.0}});

    CHECK_THROWS_AS(p.fit(Matrix(0, 2), Matrix(0, 1), 1), std::invalid_argument);
    CHECK_THROWS_AS(p.fit(X, Matrix::from_rows({{1.0}}), 1), std::invalid_argument);
    CHECK_THROWS_AS(p.fit(Matrix::from_rows({{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}}), y, 1), std::invalid_argument);
    CHECK_THROWS_AS(p.fit(X, y, 0), std::invalid_argument);
    CHECK_THROWS_AS(p.fit(X, y, -3), std::invalid_argument);
    CHECK_NOTHROW(p.fit(X, y, 1));
}

TEST_CASE("loss history length equals epochs") {
    std::mt19937_64 rng(42);
    const tp3::Dataset data = tp3::and_dataset();
    SimplePerceptron p(2, "step", 0.1, rng);
    CHECK(p.fit(data.X, data.y, 7).loss_per_epoch.size() == 7);
}

TEST_CASE("flat_weights excludes the bias") {
    std::mt19937_64 rng(3);
    SimplePerceptron p(3, "tanh", 0.1, rng);
    CHECK(p.flat_weights().size() == 3);
    CHECK(p.n_inputs() == 3);
}
