#include "mlp.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace tp3 {

MLP::MLP(const std::vector<std::size_t>& layer_sizes, const std::string& activation_name,
         double learning_rate, std::mt19937_64& rng)
    : MLP(layer_sizes, activation_name, learning_rate, rng,
          std::make_unique<MseLoss>(),
          std::make_unique<SgdOptimizer>(learning_rate),
          false) {}

MLP::MLP(const std::vector<std::size_t>& layer_sizes, const std::string& activation_name,
         double learning_rate, std::mt19937_64& rng,
         std::unique_ptr<Loss> loss,
         std::unique_ptr<Optimizer> optimizer,
         bool use_softmax_output)
    : layer_sizes_(layer_sizes),
      activation_(activation_by_name(activation_name)),
      learning_rate_(learning_rate),
      loss_(std::move(loss)),
      optimizer_(std::move(optimizer)),
      use_softmax_output_(use_softmax_output) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("MLP: layer_sizes must contain at least 2 layers, got " +
                                    std::to_string(layer_sizes.size()));
    }
    for (std::size_t i = 0; i < layer_sizes.size(); ++i) {
        if (layer_sizes[i] == 0) {
            throw std::invalid_argument("MLP: layer " + std::to_string(i) + " has size 0");
        }
    }
    if (!std::isfinite(learning_rate) || !(learning_rate > 0.0)) {
        throw std::invalid_argument("MLP: learning_rate must be a positive finite number, got " +
                                    std::to_string(learning_rate));
    }
    if (!loss_) {
        throw std::invalid_argument("MLP: loss cannot be null");
    }
    if (!optimizer_) {
        throw std::invalid_argument("MLP: optimizer cannot be null");
    }

    const bool is_cross_entropy = (std::string(loss_->name()) == "cross_entropy");
    if (is_cross_entropy != use_softmax_output_) {
        throw std::invalid_argument("MLP: loss '" + std::string(loss_->name()) +
                                    "' is incompatible with use_softmax_output=" +
                                    (use_softmax_output_ ? "true" : "false"));
    }

    const std::size_t num_weight_layers = layer_sizes_.size() - 1;
    weights_.reserve(num_weight_layers);
    biases_.reserve(num_weight_layers);

    for (std::size_t l = 0; l < num_weight_layers; ++l) {
        weights_.push_back(Matrix::random(layer_sizes_[l], layer_sizes_[l + 1], rng, -0.5, 0.5));
        biases_.push_back(Matrix::random(1, layer_sizes_[l + 1], rng, -0.5, 0.5));
    }
}

void MLP::set_weights_and_biases(const std::vector<Matrix>& weights, const std::vector<Matrix>& biases) {
    const std::size_t expected_layers = layer_sizes_.size() - 1;
    if (weights.size() != expected_layers) {
        throw std::invalid_argument("set_weights_and_biases: expected " + std::to_string(expected_layers) +
                                    " weight matrices, got " + std::to_string(weights.size()));
    }
    if (biases.size() != expected_layers) {
        throw std::invalid_argument("set_weights_and_biases: expected " + std::to_string(expected_layers) +
                                    " bias matrices, got " + std::to_string(biases.size()));
    }
    for (std::size_t l = 0; l < expected_layers; ++l) {
        if (weights[l].rows() != layer_sizes_[l] || weights[l].cols() != layer_sizes_[l + 1]) {
            throw std::invalid_argument("set_weights_and_biases: layer " + std::to_string(l) +
                                        " weight shape mismatch, expected " + std::to_string(layer_sizes_[l]) +
                                        "x" + std::to_string(layer_sizes_[l + 1]) + ", got " +
                                        std::to_string(weights[l].rows()) + "x" +
                                        std::to_string(weights[l].cols()));
        }
        if (biases[l].rows() != 1 || biases[l].cols() != layer_sizes_[l + 1]) {
            throw std::invalid_argument("set_weights_and_biases: layer " + std::to_string(l) +
                                        " bias shape mismatch, expected 1x" +
                                        std::to_string(layer_sizes_[l + 1]) + ", got " +
                                        std::to_string(biases[l].rows()) + "x" +
                                        std::to_string(biases[l].cols()));
        }
    }
    weights_ = weights;
    biases_ = biases;
}

TrainResult MLP::fit(const Matrix& X, const Matrix& y, int epochs,
                    std::function<void(int epoch, double loss)> on_epoch) {
    if (X.rows() == 0) {
        throw std::invalid_argument("fit: X has zero rows");
    }
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("fit: X has " + std::to_string(X.rows()) + " rows but y has " +
                                    std::to_string(y.rows()));
    }
    if (X.cols() != layer_sizes_.front()) {
        throw std::invalid_argument("fit: X has " + std::to_string(X.cols()) + " columns, expected " +
                                    std::to_string(layer_sizes_.front()));
    }
    if (y.cols() != layer_sizes_.back()) {
        throw std::invalid_argument("fit: y has " + std::to_string(y.cols()) + " columns, expected " +
                                    std::to_string(layer_sizes_.back()));
    }
    if (epochs <= 0) {
        throw std::invalid_argument("fit: epochs must be > 0, got " + std::to_string(epochs));
    }

    TrainResult result;
    result.loss_per_epoch.reserve(static_cast<std::size_t>(epochs));

    const std::size_t num_weight_layers = layer_sizes_.size() - 1;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (std::size_t i = 0; i < X.rows(); ++i) {
            // Forward pass: compute activations layer by layer using Matrix products.
            std::vector<Matrix> A;
            std::vector<Matrix> H;
            A.reserve(layer_sizes_.size());
            H.reserve(num_weight_layers);

            Matrix a = X.row(i);
            A.push_back(a);

            for (std::size_t l = 0; l < num_weight_layers; ++l) {
                Matrix h = (a * weights_[l]) + biases_[l];
                H.push_back(h);
                if (l == num_weight_layers - 1 && use_softmax_output_) {
                    a = softmax_rows(h);
                } else {
                    a = h.apply(activation_.f);
                }
                A.push_back(a);
            }

            // Backward pass: compute deltas via matrix transpose and product.
            std::vector<Matrix> deltas(num_weight_layers);

            // Output layer delta
            const Matrix target = y.row(i);
            deltas[num_weight_layers - 1] = loss_->output_delta(
                target, A.back(), H.back(), use_softmax_output_ ? nullptr : activation_.df);

            // Hidden layer deltas backward
            for (std::size_t l = num_weight_layers - 1; l > 0; --l) {
                const std::size_t prev = l - 1;
                deltas[prev] = (deltas[l] * weights_[l].transpose()).hadamard(H[prev].apply(activation_.df));
            }

            // Parameter updates via Optimizer
            for (std::size_t l = 0; l < num_weight_layers; ++l) {
                optimizer_->update(weights_[l], A[l].transpose() * deltas[l], 2 * l);
                optimizer_->update(biases_[l], deltas[l], 2 * l + 1);
            }
        }

        const double current_loss = loss_->compute(y, predict(X));
        result.loss_per_epoch.push_back(current_loss);
        if (on_epoch) {
            on_epoch(epoch + 1, current_loss);
        }
    }

    return result;
}

std::vector<Matrix> MLP::layer_activations(const Matrix& X) const {
    if (X.cols() != layer_sizes_.front()) {
        throw std::invalid_argument("predict: X has " + std::to_string(X.cols()) + " columns, expected " +
                                    std::to_string(layer_sizes_.front()));
    }

    std::vector<Matrix> layers;
    layers.reserve(layer_sizes_.size());
    layers.push_back(X);

    const std::size_t num_weight_layers = layer_sizes_.size() - 1;
    for (std::size_t l = 0; l < num_weight_layers; ++l) {
        layers.emplace_back(X.rows(), layer_sizes_[l + 1]);
    }

    for (std::size_t i = 0; i < X.rows(); ++i) {
        Matrix a = X.row(i);
        for (std::size_t l = 0; l < num_weight_layers; ++l) {
            Matrix h = (a * weights_[l]) + biases_[l];
            if (l == num_weight_layers - 1 && use_softmax_output_) {
                a = softmax_rows(h);
            } else {
                a = h.apply(activation_.f);
            }
            for (std::size_t j = 0; j < layer_sizes_[l + 1]; ++j) {
                layers[l + 1](i, j) = a(0, j);
            }
        }
    }

    return layers;
}

Matrix MLP::predict(const Matrix& X) const {
    return layer_activations(X).back();
}

std::vector<double> MLP::flat_weights() const {
    std::vector<double> out;
    const std::size_t num_weight_layers = layer_sizes_.size() - 1;
    for (std::size_t l = 0; l < num_weight_layers; ++l) {
        for (double w : weights_[l].data()) {
            out.push_back(w);
        }
        for (double b : biases_[l].data()) {
            out.push_back(b);
        }
    }
    return out;
}

}  // namespace tp3
