#pragma once

#include <memory>
#include <string>
#include <vector>

#include "matrix.hpp"

namespace tp3 {

// Abstract base class for loss functions.
class Loss {
public:
    virtual ~Loss() = default;

    // Evaluates the loss between target and predicted values.
    virtual double compute(const Matrix& target, const Matrix& predicted) const = 0;

    // Computes the output layer delta (dL/dh or combined dL/d(preactivation)).
    // For MSE: (target - predicted) * df(preactivation). Throws if df is nullptr.
    // For CrossEntropy with Softmax: target - predicted. Throws if df is non-null.
    virtual Matrix output_delta(const Matrix& target, const Matrix& predicted,
                                const Matrix& preactivation,
                                double (*output_activation_df)(double)) const = 0;

    virtual const char* name() const = 0;
};

class MseLoss : public Loss {
public:
    double compute(const Matrix& target, const Matrix& predicted) const override;

    Matrix output_delta(const Matrix& target, const Matrix& predicted,
                        const Matrix& preactivation,
                        double (*output_activation_df)(double)) const override;

    const char* name() const override { return "mse"; }
};

class CrossEntropyLoss : public Loss {
public:
    double compute(const Matrix& target, const Matrix& predicted) const override;

    Matrix output_delta(const Matrix& target, const Matrix& predicted,
                        const Matrix& preactivation,
                        double (*output_activation_df)(double)) const override;

    const char* name() const override { return "cross_entropy"; }
};

const std::vector<std::string>& loss_names();

// Case-sensitive exact lookup. Throws std::invalid_argument naming the allowed set if not found.
std::unique_ptr<Loss> loss_by_name(const std::string& name);

}  // namespace tp3
