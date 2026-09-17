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

// One neuron: h = w . x + b, o = f(h). The activation name selects the
// step / linear (identity) / non-linear (tanh, sigmoid) variant; all share
// the same online gradient-descent update.
class SimplePerceptron : public Model {
public:
    // Backward-compatible constructor: MSE loss and SGD optimizer.
    SimplePerceptron(std::size_t n_inputs, const std::string& activation_name, double learning_rate,
                     std::mt19937_64& rng);

    // Full constructor accepting custom loss and optimizer.
    SimplePerceptron(std::size_t n_inputs, const std::string& activation_name, double learning_rate,
                     std::mt19937_64& rng, std::unique_ptr<Loss> loss,
                     std::unique_ptr<Optimizer> optimizer);

    // Overrides the random initialisation; w must be n_inputs x 1.
    void set_weights(const Matrix& w, double bias);

    // Online GD in dataset order, no shuffling; loss = loss_->compute over X after each epoch.
    TrainResult fit(const Matrix& X, const Matrix& y, int epochs,
                    std::function<void(int epoch, double loss)> on_epoch = nullptr) override;

    // ((X * w) + b).apply(f), N x 1.
    Matrix predict(const Matrix& X) const override;

    const Matrix& weights() const { return w_; }
    double bias() const { return bias_(0, 0); }
    const Activation& activation() const { return activation_; }
    double learning_rate() const { return learning_rate_; }
    std::size_t n_inputs() const { return n_inputs_; }
    const Loss& loss() const { return *loss_; }
    const Optimizer& optimizer() const { return *optimizer_; }

    // The n_inputs weights; bias excluded.
    std::vector<double> flat_weights() const override;

private:
    std::size_t n_inputs_;
    Activation activation_;
    double learning_rate_;
    std::unique_ptr<Loss> loss_;
    std::unique_ptr<Optimizer> optimizer_;
    Matrix w_;
    Matrix bias_;  // 1x1 matrix
};

}  // namespace tp3
