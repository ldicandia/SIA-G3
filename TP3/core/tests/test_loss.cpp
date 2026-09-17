#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "activations.hpp"
#include "doctest.h"
#include "loss.hpp"
#include "matrix.hpp"
#include "training.hpp"

using tp3::CrossEntropyLoss;
using tp3::Loss;
using tp3::Matrix;
using tp3::MseLoss;
using tp3::activation_by_name;
using tp3::loss_by_name;
using tp3::loss_names;
using tp3::mean_squared_error;
using tp3::softmax_rows;

TEST_CASE("MseLoss: compute and output_delta behavior") {
    MseLoss mse;
    CHECK(std::string(mse.name()) == "mse");

    Matrix target = Matrix::from_rows({{1.0, 0.0}, {0.5, 0.8}});
    Matrix pred = Matrix::from_rows({{0.8, 0.2}, {0.6, 0.7}});

    CHECK(mse.compute(target, pred) == doctest::Approx(mean_squared_error(target, pred)));

    const tp3::Activation act = activation_by_name("tanh");
    Matrix preact = Matrix::from_rows({{0.5, -0.5}, {0.2, 0.1}});
    Matrix delta = mse.output_delta(target, pred, preact, act.df);

    Matrix expected = (target - pred).hadamard(preact.apply(act.df));
    CHECK(delta == expected);

    CHECK_THROWS_AS(mse.output_delta(target, pred, preact, nullptr), std::invalid_argument);
}

TEST_CASE("CrossEntropyLoss: compute, output_delta, and saturation safety") {
    CrossEntropyLoss ce;
    CHECK(std::string(ce.name()) == "cross_entropy");

    Matrix logits = Matrix::from_rows({{1.0, 2.0, 3.0}});
    Matrix target = Matrix::from_rows({{0.0, 0.0, 1.0}});
    Matrix predicted = softmax_rows(logits);

    // Hand-verified values from requirements:
    // Loss should equal 0.4076059644 within 1e-6
    CHECK(ce.compute(target, predicted) == doctest::Approx(0.4076059644).epsilon(1e-6));

    // output_delta should equal target - predicted = [-0.0900305732, -0.2447284711, 0.3347590442]
    Matrix delta = ce.output_delta(target, predicted, logits, nullptr);
    CHECK(delta.rows() == 1);
    CHECK(delta.cols() == 3);
    CHECK(delta(0, 0) == doctest::Approx(-0.0900305732).epsilon(1e-6));
    CHECK(delta(0, 1) == doctest::Approx(-0.2447284711).epsilon(1e-6));
    CHECK(delta(0, 2) == doctest::Approx(0.3347590442).epsilon(1e-6));

    // Throws if df is non-null
    const tp3::Activation act = activation_by_name("identity");
    CHECK_THROWS_AS(ce.output_delta(target, predicted, logits, act.df), std::invalid_argument);

    // Saturation test: probability exactly 0.0 or 1.0 must not produce NaN/Inf
    Matrix saturated_pred = Matrix::from_rows({{0.0, 1.0}});
    Matrix sat_target = Matrix::from_rows({{1.0, 0.0}});
    double sat_loss = ce.compute(sat_target, saturated_pred);
    CHECK_FALSE(std::isnan(sat_loss));
    CHECK_FALSE(std::isinf(sat_loss));
    CHECK(sat_loss > 0.0);
}

TEST_CASE("loss_by_name: case-sensitive lookup, valid and invalid sets") {
    auto mse = loss_by_name("mse");
    CHECK(mse != nullptr);
    CHECK(std::string(mse->name()) == "mse");

    auto ce = loss_by_name("cross_entropy");
    CHECK(ce != nullptr);
    CHECK(std::string(ce->name()) == "cross_entropy");

    CHECK_THROWS_AS(loss_by_name("MSE"), std::invalid_argument);
    CHECK_THROWS_AS(loss_by_name(""), std::invalid_argument);
    CHECK_THROWS_AS(loss_by_name("bogus"), std::invalid_argument);

    const std::vector<std::string> expected = {"mse", "cross_entropy"};
    CHECK(loss_names() == expected);
}
