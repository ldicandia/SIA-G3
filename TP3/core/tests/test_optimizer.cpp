#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"
#include "matrix.hpp"
#include "optimizer.hpp"

using tp3::AdamOptimizer;
using tp3::Matrix;
using tp3::MomentumOptimizer;
using tp3::Optimizer;
using tp3::SgdOptimizer;
using tp3::optimizer_by_name;
using tp3::optimizer_names;

TEST_CASE("SgdOptimizer: update arithmetic matches param + grad * lr") {
    SgdOptimizer opt(0.1);
    CHECK(std::string(opt.name()) == "sgd");

    Matrix param = Matrix::from_rows({{1.0, 2.0}});
    Matrix grad = Matrix::from_rows({{0.5, -0.5}});

    opt.update(param, grad, 0);

    CHECK(param(0, 0) == doctest::Approx(1.05));
    CHECK(param(0, 1) == doctest::Approx(1.95));
}

TEST_CASE("MomentumOptimizer: tracks velocity per param_id independently") {
    MomentumOptimizer opt(0.1, 0.9);
    CHECK(std::string(opt.name()) == "momentum");

    Matrix param0 = Matrix::from_rows({{0.0}});
    Matrix param1 = Matrix::from_rows({{0.0}});
    Matrix grad = Matrix::from_rows({{1.0}});

    // Step 1 for id 0: velocity = 0 * 0.9 + 1.0 * 0.1 = 0.1 -> param0 = 0.1
    opt.update(param0, grad, 0);
    CHECK(param0(0, 0) == doctest::Approx(0.1));

    // Step 2 for id 0: velocity = 0.1 * 0.9 + 1.0 * 0.1 = 0.19 -> param0 = 0.29
    opt.update(param0, grad, 0);
    CHECK(param0(0, 0) == doctest::Approx(0.29));

    // Step 1 for id 1 (must be independent of id 0!): velocity = 0.1 -> param1 = 0.1
    opt.update(param1, grad, 1);
    CHECK(param1(0, 0) == doctest::Approx(0.1));
}

TEST_CASE("AdamOptimizer: applies bias-corrected first and second moments per param_id") {
    AdamOptimizer opt(0.001, 0.9, 0.999, 1e-8);
    CHECK(std::string(opt.name()) == "adam");

    Matrix param = Matrix::from_rows({{1.0, -1.0}});
    Matrix grad = Matrix::from_rows({{0.1, -0.2}});

    opt.update(param, grad, 0);

    // Initial param changed
    CHECK(param(0, 0) != 1.0);
    CHECK(param(0, 1) != -1.0);

    // Another param with different id has separate state
    Matrix other_param = Matrix::from_rows({{0.0}});
    Matrix other_grad = Matrix::from_rows({{1.0}});
    opt.update(other_param, other_grad, 1);
    CHECK(other_param(0, 0) > 0.0);
}

TEST_CASE("optimizer_by_name: case-sensitive lookup, valid and invalid sets") {
    auto sgd = optimizer_by_name("sgd", 0.1);
    CHECK(sgd != nullptr);
    CHECK(std::string(sgd->name()) == "sgd");

    auto momentum = optimizer_by_name("momentum", 0.1, 0.9);
    CHECK(momentum != nullptr);
    CHECK(std::string(momentum->name()) == "momentum");

    auto adam = optimizer_by_name("adam", 0.001);
    CHECK(adam != nullptr);
    CHECK(std::string(adam->name()) == "adam");

    CHECK_THROWS_AS(optimizer_by_name("Adam", 0.1), std::invalid_argument);
    CHECK_THROWS_AS(optimizer_by_name("", 0.1), std::invalid_argument);
    CHECK_THROWS_AS(optimizer_by_name("rmsprop", 0.1), std::invalid_argument);

    const std::vector<std::string> expected = {"sgd", "momentum", "adam"};
    CHECK(optimizer_names() == expected);
}
