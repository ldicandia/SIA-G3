#pragma once

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "activations.hpp"
#include "matrix.hpp"
#include "model.hpp"
#include "training.hpp"

namespace tp3 {

// Multilayer Perceptron of arbitrary layer sizes (e.g. {2, 2, 1}, {2, 3, 2, 1}).
// Computes forward and backward passes strictly via Matrix operations.
class MLP : public Model {
public:
    // layer_sizes: dimensions for [input, hidden..., output]
    // For l = 0 .. layer_sizes.size() - 2:
    //   W[l] is layer_sizes[l] x layer_sizes[l+1], uniform in [-0.5, 0.5]
    //   b[l] is 1 x layer_sizes[l+1], uniform in [-0.5, 0.5]
    MLP(const std::vector<std::size_t>& layer_sizes, const std::string& activation_name,
        double learning_rate, std::mt19937_64& rng);

    // Explicitly sets weights and biases (for tests, hand-calculation reproduction).
    // Throws std::invalid_argument if counts or dimensions mismatch layer_sizes.
    void set_weights_and_biases(const std::vector<Matrix>& weights, const std::vector<Matrix>& biases);

    // Online gradient descent across dataset in row order, no shuffling.
    // Loss per epoch is MSE over the full dataset after each epoch.
    TrainResult fit(const Matrix& X, const Matrix& y, int epochs) override;

    // Forward pass for N x n_inputs matrix X, returning N x n_outputs.
    Matrix predict(const Matrix& X) const override;

    // All weights and biases concatenated in deterministic order:
    // W[0].data(), b[0].data(), W[1].data(), b[1].data(), ...
    std::vector<double> flat_weights() const override;

    const std::vector<Matrix>& weights() const { return weights_; }
    const std::vector<Matrix>& biases() const { return biases_; }
    const std::vector<std::size_t>& layer_sizes() const { return layer_sizes_; }
    const Activation& activation() const { return activation_; }
    double learning_rate() const { return learning_rate_; }

private:
    std::vector<std::size_t> layer_sizes_;
    Activation activation_;
    double learning_rate_;
    std::vector<Matrix> weights_;  // L-1 weight matrices
    std::vector<Matrix> biases_;   // L-1 bias row vectors (1 x layer_sizes[l+1])
};

}  // namespace tp3
