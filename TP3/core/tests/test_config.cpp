#include <stdexcept>
#include <string>

#include "config.hpp"
#include "doctest.h"

using tp3::RunConfig;

TEST_CASE("RunConfig::parse: valid config and key reordering invariance") {
    std::string json1 = "{\n"
                        "  \"format_version\": 1,\n"
                        "  \"model_type\": \"mlp\",\n"
                        "  \"layer_sizes\": [2, 2, 1],\n"
                        "  \"activation\": \"tanh\",\n"
                        "  \"use_softmax_output\": false,\n"
                        "  \"loss\": \"mse\",\n"
                        "  \"optimizer\": \"sgd\",\n"
                        "  \"learning_rate\": 0.1,\n"
                        "  \"epochs\": 100,\n"
                        "  \"seed\": 42,\n"
                        "  \"dataset_kind\": \"validation\",\n"
                        "  \"dataset_case\": \"xor\",\n"
                        "  \"output_dir\": \"runs\",\n"
                        "  \"progress_interval_epochs\": 50\n"
                        "}";

    std::string json2 = "{\n"
                        "  \"progress_interval_epochs\": 50,\n"
                        "  \"dataset_case\": \"xor\",\n"
                        "  \"dataset_kind\": \"validation\",\n"
                        "  \"seed\": 42,\n"
                        "  \"epochs\": 100,\n"
                        "  \"learning_rate\": 0.1,\n"
                        "  \"optimizer\": \"sgd\",\n"
                        "  \"loss\": \"mse\",\n"
                        "  \"use_softmax_output\": false,\n"
                        "  \"activation\": \"tanh\",\n"
                        "  \"layer_sizes\": [2, 2, 1],\n"
                        "  \"model_type\": \"mlp\",\n"
                        "  \"output_dir\": \"runs\",\n"
                        "  \"format_version\": 1\n"
                        "}";

    RunConfig c1 = RunConfig::parse(json1);
    RunConfig c2 = RunConfig::parse(json2);

    CHECK(c1.format_version == c2.format_version);
    CHECK(c1.model_type == c2.model_type);
    CHECK(c1.layer_sizes == c2.layer_sizes);
    CHECK(c1.activation == c2.activation);
    CHECK(c1.use_softmax_output == c2.use_softmax_output);
    CHECK(c1.loss == c2.loss);
    CHECK(c1.optimizer_name == c2.optimizer_name);
    CHECK(c1.learning_rate == c2.learning_rate);
    CHECK(c1.epochs == c2.epochs);
    CHECK(c1.seed == c2.seed);
    CHECK(c1.dataset_kind == c2.dataset_kind);
    CHECK(c1.dataset_case == c2.dataset_case);
    CHECK(c1.output_dir == c2.output_dir);
    CHECK(c1.progress_interval_epochs == c2.progress_interval_epochs);
}

TEST_CASE("RunConfig::parse: missing required key throws with key name") {
    std::string json_missing_lr = "{\n"
                                  "  \"format_version\": 1,\n"
                                  "  \"model_type\": \"mlp\",\n"
                                  "  \"layer_sizes\": [2, 2, 1],\n"
                                  "  \"activation\": \"tanh\",\n"
                                  "  \"loss\": \"mse\",\n"
                                  "  \"optimizer\": \"sgd\",\n"
                                  "  \"epochs\": 100,\n"
                                  "  \"seed\": 42,\n"
                                  "  \"dataset_kind\": \"validation\",\n"
                                  "  \"dataset_case\": \"xor\"\n"
                                  "}";

    try {
        RunConfig::parse(json_missing_lr);
        CHECK(false);  // Should have thrown
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        CHECK(msg.find("learning_rate") != std::string::npos);
    }
}

TEST_CASE("RunConfig::parse: invalid values and unsupported format_version") {
    CHECK_THROWS_AS(RunConfig::parse("{\"format_version\": 2, \"model_type\": \"mlp\"}"), std::invalid_argument);

    std::string base = "{\n"
                       "  \"format_version\": 1,\n"
                       "  \"model_type\": \"mlp\",\n"
                       "  \"layer_sizes\": [2, 1],\n"
                       "  \"activation\": \"tanh\",\n"
                       "  \"loss\": \"mse\",\n"
                       "  \"optimizer\": \"sgd\",\n"
                       "  \"learning_rate\": 0.1,\n"
                       "  \"epochs\": 10,\n"
                       "  \"seed\": 42,\n"
                       "  \"dataset_kind\": \"validation\",\n"
                       "  \"dataset_case\": \"xor\"\n"
                       "}";

    // Unrecognized model_type
    std::string bad_model = base;
    bad_model.replace(bad_model.find("\"mlp\""), 5, "\"bogus\"");
    CHECK_THROWS_AS(RunConfig::parse(bad_model), std::invalid_argument);

    // Unrecognized loss
    std::string bad_loss = base;
    bad_loss.replace(bad_loss.find("\"mse\""), 5, "\"cross_entropy_bogus\"");
    CHECK_THROWS_AS(RunConfig::parse(bad_loss), std::invalid_argument);

    // Unrecognized optimizer
    std::string bad_opt = base;
    bad_opt.replace(bad_opt.find("\"sgd\""), 5, "\"rmsprop\"");
    CHECK_THROWS_AS(RunConfig::parse(bad_opt), std::invalid_argument);

    // Unused optimizer hyperparameter does not throw
    std::string opt_extra = "{\n"
                            "  \"format_version\": 1,\n"
                            "  \"model_type\": \"mlp\",\n"
                            "  \"layer_sizes\": [2, 1],\n"
                            "  \"activation\": \"tanh\",\n"
                            "  \"loss\": \"mse\",\n"
                            "  \"optimizer\": \"sgd\",\n"
                            "  \"adam_beta1\": 0.85,\n"
                            "  \"learning_rate\": 0.1,\n"
                            "  \"epochs\": 10,\n"
                            "  \"seed\": 42,\n"
                            "  \"dataset_kind\": \"validation\",\n"
                            "  \"dataset_case\": \"xor\"\n"
                            "}";
    RunConfig cfg = RunConfig::parse(opt_extra);
    CHECK(cfg.optimizer_name == "sgd");
    CHECK(cfg.adam_beta1 == doctest::Approx(0.85));
}
