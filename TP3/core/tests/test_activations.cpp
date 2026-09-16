// ENG-01 / VAL-08: every activation's f and df against closed-form values,
// byte-exact lookup by name, and the fixed activation_names() order.
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "activations.hpp"
#include "doctest.h"

using tp3::Activation;
using tp3::activation_by_name;
using tp3::activation_names;

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
