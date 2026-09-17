#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "config.hpp"
#include "matrix.hpp"

namespace tp3 {

// Complete model snapshot paired with its RunConfig for persistence and training resumption.
struct ModelSnapshot {
    std::string model_type;  // "mlp" or "perceptron"
    std::vector<std::size_t> layer_sizes;
    std::string activation;
    bool use_softmax_output = false;
    std::vector<double> flat_weights;
    RunConfig run_config;
    int epochs_completed = 0;
    std::vector<double> loss_per_epoch_so_far;
};

// Writes a snapshot atomically using a temporary file and atomic rename.
std::filesystem::path save_model(const ModelSnapshot& snapshot, const std::filesystem::path& path);

// Reads and validates a snapshot from disk.
// Throws std::invalid_argument on unrecognized model type or parameter shape mismatch.
ModelSnapshot load_model(const std::filesystem::path& path);

// Reconstructs per-layer weight and bias matrices from the standard flat parameter vector.
// Ordering: W[0], b[0], W[1], b[1], ...
void unflatten_into(const std::vector<double>& flat_weights,
                    const std::vector<std::size_t>& layer_sizes,
                    std::vector<Matrix>& weights_out,
                    std::vector<Matrix>& biases_out);

}  // namespace tp3
