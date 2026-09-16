#include "perceptron.hpp"

#include <stdexcept>

namespace tp3 {

SimplePerceptron::SimplePerceptron(std::size_t n_inputs, const std::string& activation_name,
                                   double learning_rate, std::mt19937_64& rng)
    : n_inputs_(n_inputs),
      activation_(activation_by_name(activation_name)),
      learning_rate_(learning_rate),
      w_(Matrix::random(n_inputs, 1, rng, -0.5, 0.5)),
      bias_(std::uniform_real_distribution<double>(-0.5, 0.5)(rng)) {
    if (n_inputs == 0) {
        throw std::invalid_argument("SimplePerceptron: n_inputs must be > 0");
    }
    if (!(learning_rate > 0.0)) {
        throw std::invalid_argument("SimplePerceptron: learning_rate must be > 0");
    }
}

void SimplePerceptron::set_weights(const Matrix& w, double bias) {
    if (w.rows() != n_inputs_ || w.cols() != 1) {
        throw std::invalid_argument("set_weights: expected " + std::to_string(n_inputs_) + "x1, got " +
                                    std::to_string(w.rows()) + "x" + std::to_string(w.cols()));
    }
    w_ = w;
    bias_ = bias;
}

TrainResult SimplePerceptron::fit(const Matrix& X, const Matrix& y, int epochs) {
    if (X.rows() == 0) {
        throw std::invalid_argument("fit: X has zero rows");
    }
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("fit: X has " + std::to_string(X.rows()) + " rows but y has " +
                                    std::to_string(y.rows()));
    }
    if (X.cols() != n_inputs_) {
        throw std::invalid_argument("fit: X has " + std::to_string(X.cols()) + " columns, expected " +
                                    std::to_string(n_inputs_));
    }
    if (y.cols() != 1) {
        throw std::invalid_argument("fit: y must have 1 column, got " + std::to_string(y.cols()));
    }
    if (epochs <= 0) {
        throw std::invalid_argument("fit: epochs must be > 0, got " + std::to_string(epochs));
    }

    TrainResult result;
    result.loss_per_epoch.reserve(static_cast<std::size_t>(epochs));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (std::size_t i = 0; i < X.rows(); ++i) {
            double h = bias_;
            for (std::size_t j = 0; j < n_inputs_; ++j) {
                h += w_(j, 0) * X(i, j);
            }
            const double o = activation_.f(h);
            const double delta = (y(i, 0) - o) * activation_.df(h);
            for (std::size_t j = 0; j < n_inputs_; ++j) {
                w_(j, 0) += learning_rate_ * delta * X(i, j);
            }
            bias_ += learning_rate_ * delta;
        }
        result.loss_per_epoch.push_back(mean_squared_error(y, predict(X)));
    }
    return result;
}

Matrix SimplePerceptron::predict(const Matrix& X) const {
    if (X.cols() != n_inputs_) {
        throw std::invalid_argument("predict: X has " + std::to_string(X.cols()) + " columns, expected " +
                                    std::to_string(n_inputs_));
    }
    return ((X * w_) + bias_).apply(activation_.f);
}

std::vector<double> SimplePerceptron::flat_weights() const {
    return w_.data();
}

}  // namespace tp3
