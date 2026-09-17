#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "activations.hpp"
#include "loss.hpp"
#include "matrix.hpp"
#include "model.hpp"
#include "optimizer.hpp"
#include "training.hpp"

namespace tp3 {

// Multilayer Perceptron of arbitrary layer sizes (e.g. {2, 2, 1}, {2, 3, 2, 1}).
// Computes forward and backward passes strictly via Matrix operations.
class MLP : public Model {
public:
    // Backward-compatible constructor: uses MSE loss, SGD optimizer, and no softmax.
    MLP(const std::vector<std::size_t>& layer_sizes, const std::string& activation_name,
        double learning_rate, std::mt19937_64& rng);

    // Full constructor accepting custom loss, optimizer, and optional softmax output.
    // Throws std::invalid_argument if use_softmax_output != (loss->name() == "cross_entropy").
    MLP(const std::vector<std::size_t>& layer_sizes, const std::string& activation_name,
        double learning_rate, std::mt19937_64& rng, std::unique_ptr<Loss> loss,
        std::unique_ptr<Optimizer> optimizer, bool use_softmax_output);

    // Explicitly sets weights and biases (for tests, hand-calculation reproduction).
    // Throws std::invalid_argument if counts or dimensions mismatch layer_sizes.
    void set_weights_and_biases(const std::vector<Matrix>& weights, const std::vector<Matrix>& biases);

    // Online gradient descent across dataset in row order, no shuffling.
    TrainResult fit(const Matrix& X, const Matrix& y, int epochs,
                    std::function<void(int epoch, double loss)> on_epoch = nullptr) override;

    // Forward pass for N x n_inputs matrix X, returning N x n_outputs.
    Matrix predict(const Matrix& X) const override;

    // Layer-by-layer activations: index 0 is input X, index L-1 is final output.
    // Pure function of X (recomputed on demand).
    std::vector<Matrix> layer_activations(const Matrix& X) const;

    // All weights and biases concatenated in deterministic order:
    // W[0].data(), b[0].data(), W[1].data(), b[1].data(), ...
    std::vector<double> flat_weights() const override;

    const std::vector<Matrix>& weights() const { return weights_; }
    const std::vector<Matrix>& biases() const { return biases_; }
    const std::vector<std::size_t>& layer_sizes() const { return layer_sizes_; }
    const Activation& activation() const { return activation_; }
    double learning_rate() const { return learning_rate_; }
    const Loss& loss() const { return *loss_; }
    const Optimizer& optimizer() const { return *optimizer_; }
    bool use_softmax_output() const { return use_softmax_output_; }

private:
    std::vector<std::size_t> layer_sizes_;
    Activation activation_;
    double learning_rate_;
    std::unique_ptr<Loss> loss_;
    std::unique_ptr<Optimizer> optimizer_;
    bool use_softmax_output_;
    std::vector<Matrix> weights_;  // L-1 weight matrices
    std::vector<Matrix> biases_;   // L-1 bias row vectors (1 x layer_sizes[l+1])
};

}  // namespace tp3
