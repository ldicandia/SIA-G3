#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tp3 {

// Complete configuration for a training run.
// Mapped to flat JSON file schema format_version=1.
struct RunConfig {
    int format_version = 1;
    std::string model_type;  // "mlp" or "perceptron"
    std::vector<std::size_t> layer_sizes;  // Required for "mlp"
    std::string activation;
    bool use_softmax_output = false;
    std::string loss = "mse";             // JSON key "loss"
    std::string optimizer_name = "sgd";   // JSON key "optimizer"
    double momentum_coefficient = 0.9;
    double adam_beta1 = 0.9;
    double adam_beta2 = 0.999;
    double adam_epsilon = 1e-8;
    double learning_rate = 0.01;
    int epochs = 100;
    unsigned long long seed = 42;

    std::string dataset_kind;  // "validation" or "csv"
    std::string dataset_case;  // "and", "linear", "tanh", "xor" (for "validation")
    std::string dataset_format; // "plain" or "digits" (for "csv")
    std::string dataset_path;
    std::string dataset_target_column;

    std::string output_dir = "runs";
    int progress_interval_epochs = 50;
    std::string resume_from;

    // Parses and validates a JSON run config. Throws std::invalid_argument on error.
    static RunConfig parse(const std::string& json_text);

    // Serializes to flat JSON string.
    std::string to_json() const;
};

}  // namespace tp3
