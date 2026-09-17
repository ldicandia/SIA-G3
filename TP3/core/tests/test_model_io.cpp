#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include "doctest.h"
#include "io/model_io.hpp"
#include "mlp.hpp"
#include "validation/datasets.hpp"

using tp3::Matrix;
using tp3::MLP;
using tp3::ModelSnapshot;
using tp3::RunConfig;
using tp3::load_model;
using tp3::save_model;
using tp3::unflatten_into;

TEST_CASE("ModelSnapshot: save, load, and bit-identical predictions roundtrip (ENG-11)") {
    std::mt19937_64 rng(42);
    MLP original_model({2, 2, 1}, "tanh", 0.1, rng);

    // Train briefly
    auto data = tp3::xor_dataset();
    original_model.fit(data.X, data.y, 10);

    // Build flat weights: W[0], b[0], W[1], b[1]
    std::vector<double> flat;
    for (std::size_t l = 0; l < original_model.weights().size(); ++l) {
        for (double w : original_model.weights()[l].data()) flat.push_back(w);
        for (double b : original_model.biases()[l].data()) flat.push_back(b);
    }

    RunConfig cfg;
    cfg.format_version = 1;
    cfg.model_type = "mlp";
    cfg.layer_sizes = {2, 2, 1};
    cfg.activation = "tanh";
    cfg.loss = "mse";
    cfg.optimizer_name = "sgd";
    cfg.learning_rate = 0.1;
    cfg.epochs = 10;
    cfg.seed = 42;
    cfg.dataset_kind = "validation";
    cfg.dataset_case = "xor";

    ModelSnapshot snapshot;
    snapshot.model_type = "mlp";
    snapshot.layer_sizes = {2, 2, 1};
    snapshot.activation = "tanh";
    snapshot.use_softmax_output = false;
    snapshot.flat_weights = flat;
    snapshot.epochs_completed = 10;
    snapshot.run_config = cfg;

    const std::filesystem::path model_path =
        std::filesystem::temp_directory_path() / "test_model_roundtrip.json";

    // Save model twice to test atomicity and no leftover .tmp
    save_model(snapshot, model_path);
    save_model(snapshot, model_path);

    std::filesystem::path tmp_path = model_path;
    tmp_path += ".tmp";
    CHECK_FALSE(std::filesystem::exists(tmp_path));
    CHECK(std::filesystem::exists(model_path));

    ModelSnapshot loaded = load_model(model_path);
    CHECK(loaded.model_type == "mlp");
    CHECK(loaded.layer_sizes == std::vector<std::size_t>{2, 2, 1});
    CHECK(loaded.activation == "tanh");
    CHECK(loaded.epochs_completed == 10);
    CHECK(loaded.run_config.epochs == 10);
    CHECK(loaded.run_config.learning_rate == doctest::Approx(0.1));
    CHECK(loaded.run_config.dataset_case == "xor");

    std::vector<Matrix> weights_reconstructed;
    std::vector<Matrix> biases_reconstructed;
    unflatten_into(loaded.flat_weights, loaded.layer_sizes, weights_reconstructed, biases_reconstructed);

    std::mt19937_64 rng2(999);
    MLP reloaded_model({2, 2, 1}, "tanh", 0.1, rng2);
    reloaded_model.set_weights_and_biases(weights_reconstructed, biases_reconstructed);

    Matrix orig_preds = original_model.predict(data.X);
    Matrix relo_preds = reloaded_model.predict(data.X);
    CHECK(orig_preds == relo_preds);

    std::filesystem::remove(model_path);
}

TEST_CASE("ModelSnapshot: rejects corrupted parameters or unknown model_type") {
    const std::filesystem::path corrupted_path =
        std::filesystem::temp_directory_path() / "corrupted_model.json";

    // Shape mismatch (too few weights)
    {
        std::ofstream f(corrupted_path);
        f << "{\n"
          << "  \"format_version\": 1,\n"
          << "  \"model_type\": \"mlp\",\n"
          << "  \"layer_sizes\": [2, 2, 1],\n"
          << "  \"activation\": \"tanh\",\n"
          << "  \"flat_weights\": [1.0, 2.0],\n"
          << "  \"loss\": \"mse\",\n"
          << "  \"optimizer\": \"sgd\",\n"
          << "  \"learning_rate\": 0.1,\n"
          << "  \"epochs\": 10,\n"
          << "  \"seed\": 42,\n"
          << "  \"dataset_kind\": \"validation\",\n"
          << "  \"dataset_case\": \"xor\"\n"
          << "}\n";
    }
    CHECK_THROWS_AS(load_model(corrupted_path), std::invalid_argument);

    // Unknown model_type
    {
        std::ofstream f(corrupted_path);
        f << "{\n"
          << "  \"format_version\": 1,\n"
          << "  \"model_type\": \"bogus\",\n"
          << "  \"layer_sizes\": [2, 1],\n"
          << "  \"activation\": \"tanh\",\n"
          << "  \"flat_weights\": [1.0, 2.0, 3.0],\n"
          << "  \"loss\": \"mse\",\n"
          << "  \"optimizer\": \"sgd\",\n"
          << "  \"learning_rate\": 0.1,\n"
          << "  \"epochs\": 10,\n"
          << "  \"seed\": 42,\n"
          << "  \"dataset_kind\": \"validation\",\n"
          << "  \"dataset_case\": \"xor\"\n"
          << "}\n";
    }
    CHECK_THROWS_AS(load_model(corrupted_path), std::invalid_argument);

    std::filesystem::remove(corrupted_path);
}
