#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "doctest.h"
#include "matrix.hpp"
#include "mlp.hpp"

using tp3::Matrix;
using tp3::MLP;

TEST_CASE("MLP constructor argument validation") {
    std::mt19937_64 rng(42);

    SUBCASE("fewer than 2 layers throws") {
        CHECK_THROWS_AS(MLP({}, "tanh", 0.1, rng), std::invalid_argument);
        CHECK_THROWS_AS(MLP({2}, "tanh", 0.1, rng), std::invalid_argument);
    }

    SUBCASE("zero-sized layer throws") {
        CHECK_THROWS_AS(MLP({0, 2, 1}, "tanh", 0.1, rng), std::invalid_argument);
        CHECK_THROWS_AS(MLP({2, 0, 1}, "tanh", 0.1, rng), std::invalid_argument);
        CHECK_THROWS_AS(MLP({2, 2, 0}, "tanh", 0.1, rng), std::invalid_argument);
    }

    SUBCASE("invalid learning rate throws") {
        CHECK_THROWS_AS(MLP({2, 2, 1}, "tanh", 0.0, rng), std::invalid_argument);
        CHECK_THROWS_AS(MLP({2, 2, 1}, "tanh", -0.1, rng), std::invalid_argument);
        CHECK_THROWS_AS(MLP({2, 2, 1}, "tanh", std::numeric_limits<double>::infinity(), rng),
                        std::invalid_argument);
        CHECK_THROWS_AS(MLP({2, 2, 1}, "tanh", std::numeric_limits<double>::quiet_NaN(), rng),
                        std::invalid_argument);
    }

    SUBCASE("unknown activation throws") {
        CHECK_THROWS_AS(MLP({2, 2, 1}, "relu", 0.1, rng), std::invalid_argument);
        CHECK_THROWS_AS(MLP({2, 2, 1}, "", 0.1, rng), std::invalid_argument);
    }
}

TEST_CASE("MLP set_weights_and_biases shape validation") {
    std::mt19937_64 rng(42);
    MLP model({2, 2, 1}, "tanh", 0.1, rng);

    SUBCASE("wrong number of weight matrices throws") {
        CHECK_THROWS_AS(model.set_weights_and_biases({}, {Matrix(1, 2), Matrix(1, 1)}),
                        std::invalid_argument);
    }

    SUBCASE("wrong weight matrix shape throws") {
        CHECK_THROWS_AS(model.set_weights_and_biases({Matrix(3, 2), Matrix(2, 1)},
                                                     {Matrix(1, 2), Matrix(1, 1)}),
                        std::invalid_argument);
    }

    SUBCASE("wrong bias matrix shape throws") {
        CHECK_THROWS_AS(model.set_weights_and_biases({Matrix(2, 2), Matrix(2, 1)},
                                                     {Matrix(2, 2), Matrix(1, 1)}),
                        std::invalid_argument);
    }
}

TEST_CASE("MLP forward pass matches analytical calculation") {
    std::mt19937_64 rng(42);
    MLP model({2, 2, 1}, "tanh", 0.1, rng);

    // Initial weights matching docs/xor_a_mano.md Section 2.1
    Matrix W0 = Matrix::from_rows({{0.15, -0.25}, {0.35, 0.45}});
    Matrix b0 = Matrix::from_rows({{0.10, -0.20}});
    Matrix W1 = Matrix::from_rows({{0.50}, {-0.40}});
    Matrix b1 = Matrix::from_rows({{0.05}});
    model.set_weights_and_biases({W0, W1}, {b0, b1});

    Matrix X = Matrix::from_rows({{-1.0, 1.0}});
    Matrix out = model.predict(X);

    CHECK(out.rows() == 1);
    CHECK(out.cols() == 1);
    // Calculated in docs/xor_a_mano.md: a2 = 0.01080902
    CHECK(out(0, 0) == doctest::Approx(0.01080902).epsilon(1e-6));
}

TEST_CASE("MLP single backprop step reproduces docs/xor_a_mano.md for [2,2,1] (DOC-03)") {
    std::mt19937_64 rng(42);
    MLP model({2, 2, 1}, "tanh", 0.1, rng);

    // Initial weights from Section 2.1
    Matrix W0 = Matrix::from_rows({{0.15, -0.25}, {0.35, 0.45}});
    Matrix b0 = Matrix::from_rows({{0.10, -0.20}});
    Matrix W1 = Matrix::from_rows({{0.50}, {-0.40}});
    Matrix b1 = Matrix::from_rows({{0.05}});
    model.set_weights_and_biases({W0, W1}, {b0, b1});

    Matrix X = Matrix::from_rows({{-1.0, 1.0}});
    Matrix y = Matrix::from_rows({{1.0}});

    // Train for exactly 1 epoch (1 sample = 1 update step)
    tp3::TrainResult train = model.fit(X, y, 1);
    CHECK(train.loss_per_epoch.size() == 1);

    const std::vector<Matrix>& weights = model.weights();
    const std::vector<Matrix>& biases = model.biases();

    // Verify W0_new
    CHECK(weights[0](0, 0) == doctest::Approx(0.10474303).epsilon(1e-6));
    CHECK(weights[0](0, 1) == doctest::Approx(-0.21888576).epsilon(1e-6));
    CHECK(weights[0](1, 0) == doctest::Approx(0.39525697).epsilon(1e-6));
    CHECK(weights[0](1, 1) == doctest::Approx(0.41888576).epsilon(1e-6));

    // Verify b0_new
    CHECK(biases[0](0, 0) == doctest::Approx(0.14525697).epsilon(1e-6));
    CHECK(biases[0](0, 1) == doctest::Approx(-0.23111424).epsilon(1e-6));

    // Verify W1_new
    CHECK(weights[1](0, 0) == doctest::Approx(0.52881301).epsilon(1e-6));
    CHECK(weights[1](1, 0) == doctest::Approx(-0.35429313).epsilon(1e-6));

    // Verify b1_new
    CHECK(biases[1](0, 0) == doctest::Approx(0.14890754).epsilon(1e-6));
}

TEST_CASE("MLP single backprop step reproduces docs/xor_a_mano.md for [2,3,2,1] (DOC-03)") {
    std::mt19937_64 rng(42);
    MLP model({2, 3, 2, 1}, "tanh", 0.1, rng);

    // Initial weights from Section 3.1
    Matrix W0 = Matrix::from_rows({{0.15, -0.20, 0.25}, {0.30, 0.10, -0.35}});
    Matrix b0 = Matrix::from_rows({{0.05, -0.10, 0.15}});
    Matrix W1 = Matrix::from_rows({{0.40, -0.30}, {-0.25, 0.35}, {0.20, 0.10}});
    Matrix b1 = Matrix::from_rows({{-0.05, 0.05}});
    Matrix W2 = Matrix::from_rows({{0.50}, {-0.45}});
    Matrix b2 = Matrix::from_rows({{0.10}});
    model.set_weights_and_biases({W0, W1, W2}, {b0, b1, b2});

    Matrix X = Matrix::from_rows({{-1.0, 1.0}});
    Matrix y = Matrix::from_rows({{1.0}});

    // Train for 1 epoch
    tp3::TrainResult train = model.fit(X, y, 1);
    CHECK(train.loss_per_epoch.size() == 1);

    const std::vector<Matrix>& weights = model.weights();
    const std::vector<Matrix>& biases = model.biases();

    // Verify W0_new
    CHECK(weights[0](0, 0) == doctest::Approx(0.11934106).epsilon(1e-6));
    CHECK(weights[0](0, 1) == doctest::Approx(-0.17410324).epsilon(1e-6));
    CHECK(weights[0](0, 2) == doctest::Approx(0.24575071).epsilon(1e-6));
    CHECK(weights[0](1, 0) == doctest::Approx(0.33065894).epsilon(1e-6));
    CHECK(weights[0](1, 1) == doctest::Approx(0.07410324).epsilon(1e-6));
    CHECK(weights[0](1, 2) == doctest::Approx(-0.34575071).epsilon(1e-6));

    // Verify b0_new
    CHECK(biases[0](0, 0) == doctest::Approx(0.08065894).epsilon(1e-6));
    CHECK(biases[0](0, 1) == doctest::Approx(-0.12589676).epsilon(1e-6));
    CHECK(biases[0](0, 2) == doctest::Approx(0.15424929).epsilon(1e-6));

    // Verify W1_new
    CHECK(weights[1](0, 0) == doctest::Approx(0.40935758).epsilon(1e-6));
    CHECK(weights[1](0, 1) == doctest::Approx(-0.30851195).epsilon(1e-6));
    CHECK(weights[1](1, 0) == doctest::Approx(-0.24064242).epsilon(1e-6));
    CHECK(weights[1](1, 1) == doctest::Approx(0.34148805).epsilon(1e-6));
    CHECK(weights[1](2, 0) == doctest::Approx(0.17999774).epsilon(1e-6));
    CHECK(weights[1](2, 1) == doctest::Approx(0.11819469).epsilon(1e-6));

    // Verify b1_new
    CHECK(biases[1](0, 0) == doctest::Approx(-0.00258992).epsilon(1e-6));
    CHECK(biases[1](0, 1) == doctest::Approx(0.00687430).epsilon(1e-6));

    // Verify W2_new
    CHECK(weights[2](0, 0) == doctest::Approx(0.48999250).epsilon(1e-6));
    CHECK(weights[2](1, 0) == doctest::Approx(-0.44830539).epsilon(1e-6));

    // Verify b2_new
    CHECK(biases[2](0, 0) == doctest::Approx(0.19586485).epsilon(1e-6));
}

TEST_CASE("MLP flat_weights order is deterministic and matches layer counts") {
    std::mt19937_64 rng(42);
    MLP model({2, 3, 2, 1}, "tanh", 0.1, rng);

    // Total elements:
    // W0: 2x3 = 6, b0: 1x3 = 3
    // W1: 3x2 = 6, b1: 1x2 = 2
    // W2: 2x1 = 2, b2: 1x1 = 1
    // Total = 6 + 3 + 6 + 2 + 2 + 1 = 20
    std::vector<double> flat = model.flat_weights();
    CHECK(flat.size() == 20);

    // Assert reproducibility with same seed
    std::mt19937_64 rng2(42);
    MLP model2({2, 3, 2, 1}, "tanh", 0.1, rng2);
    CHECK(flat == model2.flat_weights());
}

#include "validation/datasets.hpp"

TEST_CASE("MLP [2,2,1] and [2,3,2,1] learn XOR at seed 42 (VAL-04, VAL-05)") {
    const tp3::Dataset data = tp3::xor_dataset();

    SUBCASE("MLP [2,2,1] classifies all 4 inputs correctly") {
        std::mt19937_64 rng(42);
        MLP model({2, 2, 1}, "tanh", 0.1, rng);
        tp3::TrainResult train = model.fit(data.X, data.y, 2000);
        Matrix pred = model.predict(data.X);
        std::size_t correct = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            double sign = pred(i, 0) >= 0.0 ? 1.0 : -1.0;
            if (sign == data.y(i, 0)) ++correct;
        }
        CHECK(correct == 4);
        CHECK(train.loss_per_epoch.back() < 0.05);
    }

    SUBCASE("MLP [2,3,2,1] classifies all 4 inputs correctly") {
        std::mt19937_64 rng(42);
        MLP model({2, 3, 2, 1}, "tanh", 0.1, rng);
        tp3::TrainResult train = model.fit(data.X, data.y, 2000);
        Matrix pred = model.predict(data.X);
        std::size_t correct = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            double sign = pred(i, 0) >= 0.0 ? 1.0 : -1.0;
            if (sign == data.y(i, 0)) ++correct;
        }
        CHECK(correct == 4);
        CHECK(train.loss_per_epoch.back() < 0.05);
    }
}

TEST_CASE("MLP loss, optimizer, and softmax validation") {
    std::mt19937_64 rng(42);

    SUBCASE("cross_entropy requires use_softmax_output=true") {
        CHECK_THROWS_AS(MLP({2, 2, 1}, "tanh", 0.1, rng,
                            std::make_unique<tp3::CrossEntropyLoss>(),
                            std::make_unique<tp3::SgdOptimizer>(0.1),
                            /*use_softmax_output=*/false),
                        std::invalid_argument);
    }

    SUBCASE("mse rejects use_softmax_output=true") {
        CHECK_THROWS_AS(MLP({2, 2, 1}, "tanh", 0.1, rng,
                            std::make_unique<tp3::MseLoss>(),
                            std::make_unique<tp3::SgdOptimizer>(0.1),
                            /*use_softmax_output=*/true),
                        std::invalid_argument);
    }

    SUBCASE("null loss or optimizer throws") {
        CHECK_THROWS_AS(MLP({2, 2, 1}, "tanh", 0.1, rng,
                            nullptr,
                            std::make_unique<tp3::SgdOptimizer>(0.1),
                            false),
                        std::invalid_argument);
        CHECK_THROWS_AS(MLP({2, 2, 1}, "tanh", 0.1, rng,
                            std::make_unique<tp3::MseLoss>(),
                            nullptr,
                            false),
                        std::invalid_argument);
    }
}

TEST_CASE("MLP layer_activations matches predict and preserves shapes (ENG-15)") {
    std::mt19937_64 rng(42);

    SUBCASE("standard non-softmax MLP") {
        MLP model({2, 3, 2, 1}, "tanh", 0.1, rng);
        Matrix X = Matrix::from_rows({{-1.0, 1.0}, {0.5, -0.5}, {1.0, 1.0}});

        auto layers = model.layer_activations(X);
        CHECK(layers.size() == 4);
        CHECK(layers[0] == X);
        CHECK(layers[1].rows() == 3);
        CHECK(layers[1].cols() == 3);
        CHECK(layers[2].rows() == 3);
        CHECK(layers[2].cols() == 2);
        CHECK(layers[3].rows() == 3);
        CHECK(layers[3].cols() == 1);
        CHECK(layers.back() == model.predict(X));
    }

    SUBCASE("softmax output MLP") {
        std::mt19937_64 rng_sm(42);
        MLP model_sm({4, 5, 3}, "tanh", 0.05, rng_sm,
                     std::make_unique<tp3::CrossEntropyLoss>(),
                     std::make_unique<tp3::SgdOptimizer>(0.05),
                     /*use_softmax_output=*/true);

        Matrix X = Matrix::from_rows({{0.1, 0.2, 0.3, 0.4}, {-0.1, -0.2, 0.5, 0.0}});
        auto layers = model_sm.layer_activations(X);
        CHECK(layers.size() == 3);
        CHECK(layers[0] == X);
        CHECK(layers.back() == model_sm.predict(X));

        Matrix pred = model_sm.predict(X);
        CHECK(pred.rows() == 2);
        CHECK(pred.cols() == 3);
        for (std::size_t r = 0; r < pred.rows(); ++r) {
            double sum = pred(r, 0) + pred(r, 1) + pred(r, 2);
            CHECK(sum == doctest::Approx(1.0).epsilon(1e-6));
        }
    }
}

TEST_CASE("Optimizer selection diverges backprop effect (SGD vs Momentum)") {
    Matrix W0 = Matrix::from_rows({{0.15, -0.25}, {0.35, 0.45}});
    Matrix b0 = Matrix::from_rows({{0.10, -0.20}});
    Matrix W1 = Matrix::from_rows({{0.50}, {-0.40}});
    Matrix b1 = Matrix::from_rows({{0.05}});

    std::mt19937_64 rng1(42);
    std::mt19937_64 rng2(42);

    MLP model_sgd({2, 2, 1}, "tanh", 0.1, rng1,
                  std::make_unique<tp3::MseLoss>(),
                  std::make_unique<tp3::SgdOptimizer>(0.1),
                  false);
    model_sgd.set_weights_and_biases({W0, W1}, {b0, b1});

    MLP model_mom({2, 2, 1}, "tanh", 0.1, rng2,
                  std::make_unique<tp3::MseLoss>(),
                  std::make_unique<tp3::MomentumOptimizer>(0.1, 0.9),
                  false);
    model_mom.set_weights_and_biases({W0, W1}, {b0, b1});

    const tp3::Dataset data = tp3::xor_dataset();

    model_sgd.fit(data.X, data.y, 5);
    model_mom.fit(data.X, data.y, 5);

    auto w_sgd = model_sgd.flat_weights();
    auto w_mom = model_mom.flat_weights();

    bool has_diff = false;
    for (std::size_t i = 0; i < w_sgd.size(); ++i) {
        if (std::abs(w_sgd[i] - w_mom[i]) > 1e-9) {
            has_diff = true;
            break;
        }
    }
    CHECK(has_diff);
}
