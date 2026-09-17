#include "perceptron.hpp"

#include <cmath>
#include <stdexcept>

namespace tp3 {

SimplePerceptron::SimplePerceptron(std::size_t n_inputs, const std::string& activation_name,
                                   double learning_rate, std::mt19937_64& rng)
    : SimplePerceptron(n_inputs, activation_name, learning_rate, rng,
                       std::make_unique<MseLoss>(),
                       std::make_unique<SgdOptimizer>(learning_rate)) {}

SimplePerceptron::SimplePerceptron(std::size_t n_inputs, const std::string& activation_name,
                                   double learning_rate, std::mt19937_64& rng,
                                   std::unique_ptr<Loss> loss,
                                   std::unique_ptr<Optimizer> optimizer)
    : n_inputs_(n_inputs),
      activation_(activation_by_name(activation_name)),
      learning_rate_(learning_rate),
      loss_(std::move(loss)),
      optimizer_(std::move(optimizer)),
      w_(Matrix::random(n_inputs, 1, rng, -0.5, 0.5)),
      bias_(1, 1, std::uniform_real_distribution<double>(-0.5, 0.5)(rng)) {
    if (n_inputs == 0) {
        throw std::invalid_argument("SimplePerceptron: n_inputs must be > 0");
    }
    if (!std::isfinite(learning_rate) || !(learning_rate > 0.0)) {
        throw std::invalid_argument("SimplePerceptron: learning_rate must be > 0");
    }
    if (!loss_) {
        throw std::invalid_argument("SimplePerceptron: loss cannot be null");
    }
    if (!optimizer_) {
        throw std::invalid_argument("SimplePerceptron: optimizer cannot be null");
    }
}

void SimplePerceptron::set_weights(const Matrix& w, double bias) {
    if (w.rows() != n_inputs_ || w.cols() != 1) {
        throw std::invalid_argument("set_weights: expected " + std::to_string(n_inputs_) + "x1, got " +
                                    std::to_string(w.rows()) + "x" + std::to_string(w.cols()));
    }
    w_ = w;
    bias_ = Matrix(1, 1, bias);
}

TrainResult SimplePerceptron::fit(const Matrix& X, const Matrix& y, int epochs,
                                  std::function<void(int epoch, double loss)> on_epoch) {
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
            Matrix x_row = X.row(i);
            Matrix h = (x_row * w_) + bias_;
            Matrix o = h.apply(activation_.f);
            Matrix target_row = y.row(i);
            Matrix delta = loss_->output_delta(target_row, o, h, activation_.df);
            optimizer_->update(w_, x_row.transpose() * delta, 0);
            optimizer_->update(bias_, delta, 1);
        }
        const double current_loss = loss_->compute(y, predict(X));
        result.loss_per_epoch.push_back(current_loss);
        if (on_epoch) {
            on_epoch(epoch + 1, current_loss);
        }
    }
    return result;
}

Matrix SimplePerceptron::predict(const Matrix& X) const {
    if (X.cols() != n_inputs_) {
        throw std::invalid_argument("predict: X has " + std::to_string(X.cols()) + " columns, expected " +
                                    std::to_string(n_inputs_));
    }
    return ((X * w_) + bias_(0, 0)).apply(activation_.f);
}

std::vector<double> SimplePerceptron::flat_weights() const {
    return w_.data();
}

}  // namespace tp3
