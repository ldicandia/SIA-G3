// ENG-01 / VAL-08: every activation's f and df against closed-form values,
// byte-exact lookup by name, and the fixed activation_names() order.
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "activations.hpp"
#include "doctest.h"
#include "matrix.hpp"

using tp3::Activation;
using tp3::Matrix;
using tp3::activation_by_name;
using tp3::activation_names;
using tp3::softmax_rows;

TEST_CASE("step: f is the bipolar sign with step(0) == +1, df is 1 everywhere") {
    const Activation a = activation_by_name("step");
    CHECK(a.f(-1.0) == -1.0);
    CHECK(a.f(0.0) == 1.0);
    CHECK(a.f(1.0) == 1.0);
    CHECK(a.f(-1e-9) == -1.0);
    CHECK(a.df(-1.0) == 1.0);
    CHECK(a.df(0.0) == 1.0);
    CHECK(a.df(1.0) == 1.0);
}

TEST_CASE("identity: f(h) == h, df == 1") {
    const Activation a = activation_by_name("identity");
    for (double h : {-1.0, 0.0, 1.0}) {
        CHECK(a.f(h) == h);
        CHECK(a.df(h) == 1.0);
    }
}

TEST_CASE("tanh: f and df = 1 - tanh(h)^2") {
    const Activation a = activation_by_name("tanh");
    CHECK(a.f(0.0) == 0.0);
    CHECK(a.f(1.0) == doctest::Approx(std::tanh(1.0)));
    CHECK(a.f(-1.0) == doctest::Approx(-std::tanh(1.0)));
    CHECK(a.df(0.0) == doctest::Approx(1.0));
    CHECK(a.df(1.0) == doctest::Approx(1.0 - std::tanh(1.0) * std::tanh(1.0)));
    CHECK(a.df(-1.0) == doctest::Approx(1.0 - std::tanh(1.0) * std::tanh(1.0)));
}

TEST_CASE("sigmoid: f and df = s(1 - s)") {
    const Activation a = activation_by_name("sigmoid");
    const double s1 = 1.0 / (1.0 + std::exp(-1.0));
    CHECK(a.f(0.0) == doctest::Approx(0.5));
    CHECK(a.f(1.0) == doctest::Approx(s1));
    CHECK(a.f(-1.0) == doctest::Approx(1.0 - s1));
    CHECK(a.df(0.0) == doctest::Approx(0.25));
    CHECK(a.df(1.0) == doctest::Approx(s1 * (1.0 - s1)));
    CHECK(a.df(-1.0) == doctest::Approx(s1 * (1.0 - s1)));
}

TEST_CASE("lookup: byte-exact names, unknown/empty/case-variant throw, fixed order") {
    CHECK_THROWS_AS(activation_by_name(""), std::invalid_argument);
    CHECK_THROWS_AS(activation_by_name("relu"), std::invalid_argument);
    CHECK_THROWS_AS(activation_by_name("Tanh"), std::invalid_argument);

    const std::vector<std::string> expected = {"step", "identity", "tanh", "sigmoid"};
    CHECK(activation_names() == expected);
    for (const std::string& n : activation_names()) {
        CHECK(std::string(activation_by_name(n).name) == n);
    }
}

TEST_CASE("softmax_rows: numerical stability for large logits and row sums equal 1") {
    Matrix logits_small = Matrix::from_rows({{0.0, 1.0, 2.0}});
    Matrix logits_large = Matrix::from_rows({{1000.0, 1001.0, 1002.0}});

    Matrix sm_small = softmax_rows(logits_small);
    Matrix sm_large = softmax_rows(logits_large);

    CHECK(sm_small.rows() == 1);
    CHECK(sm_small.cols() == 3);
    CHECK(sm_small(0, 0) == doctest::Approx(0.0900305732).epsilon(1e-6));
    CHECK(sm_small(0, 1) == doctest::Approx(0.2447284711).epsilon(1e-6));
    CHECK(sm_small(0, 2) == doctest::Approx(0.6652409558).epsilon(1e-6));

    CHECK(sm_large(0, 0) == doctest::Approx(0.0900305732).epsilon(1e-6));
    CHECK(sm_large(0, 1) == doctest::Approx(0.2447284711).epsilon(1e-6));
    CHECK(sm_large(0, 2) == doctest::Approx(0.6652409558).epsilon(1e-6));

    CHECK(sm_large(0, 0) == doctest::Approx(sm_small(0, 0)));
    CHECK(sm_large(0, 1) == doctest::Approx(sm_small(0, 1)));
    CHECK(sm_large(0, 2) == doctest::Approx(sm_small(0, 2)));

    CHECK_FALSE(std::isnan(sm_large(0, 0)));
    CHECK_FALSE(std::isinf(sm_large(0, 0)));

    Matrix empty_m;
    CHECK_THROWS_AS(softmax_rows(empty_m), std::invalid_argument);
}
