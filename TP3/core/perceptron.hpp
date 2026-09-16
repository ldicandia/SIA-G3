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

// One neuron: h = w . x + b, o = f(h). The activation name selects the
// step / linear (identity) / non-linear (tanh, sigmoid) variant; all share
// the same online gradient-descent update.
class SimplePerceptron : public Model {
public:
    // Draws weights (n_inputs x 1, uniform in [-0.5, 0.5]) then the bias (one draw) from rng.
    SimplePerceptron(std::size_t n_inputs, const std::string& activation_name, double learning_rate,
                     std::mt19937_64& rng);

    // Overrides the random initialisation; w must be n_inputs x 1.
    void set_weights(const Matrix& w, double bias);

    // Online GD in dataset order, no shuffling; loss = MSE over X after each epoch.
    TrainResult fit(const Matrix& X, const Matrix& y, int epochs) override;

    // ((X * w) + b).apply(f), N x 1.
    Matrix predict(const Matrix& X) const override;

    const Matrix& weights() const { return w_; }
    double bias() const { return bias_; }
    const Activation& activation() const { return activation_; }
    double learning_rate() const { return learning_rate_; }
    std::size_t n_inputs() const { return n_inputs_; }

    // The n_inputs weights; bias excluded.
    std::vector<double> flat_weights() const override;

private:
    std::size_t n_inputs_;
    Activation activation_;
    double learning_rate_;
    Matrix w_;
    double bias_;
};

}  // namespace tp3
