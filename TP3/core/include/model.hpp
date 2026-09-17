#pragma once

#include <functional>
#include <vector>

#include "matrix.hpp"
#include "training.hpp"

namespace tp3 {

// The contract shared by the simple perceptron and the MLP:
// main.cpp, the run-JSON writer, the tests and the plotter only ever see this.
class Model {
public:
    virtual ~Model() = default;

    // Trains for exactly `epochs` epochs; returns the loss after each one.
    // Optional on_epoch callback called as on_epoch(epoch, loss) where epoch is 1-based.
    virtual TrainResult fit(const Matrix& X, const Matrix& y, int epochs,
                            std::function<void(int epoch, double loss)> on_epoch = nullptr) = 0;

    // N x outputs predictions for N x inputs.
    virtual Matrix predict(const Matrix& X) const = 0;

    // All trainable weights flattened in a model-defined, deterministic order.
    virtual std::vector<double> flat_weights() const = 0;
};

}  // namespace tp3
