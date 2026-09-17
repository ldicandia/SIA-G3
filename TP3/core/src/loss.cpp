#include "loss.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "training.hpp"

namespace tp3 {

namespace {

const char* const kLossNames[] = {
    "mse",
    "cross_entropy",
};

std::string joined_loss_names() {
    std::string out;
    for (const char* name : kLossNames) {
        if (!out.empty()) {
            out += ", ";
        }
        out += name;
    }
    return out;
}

}  // namespace

double MseLoss::compute(const Matrix& target, const Matrix& predicted) const {
    return mean_squared_error(target, predicted);
}

Matrix MseLoss::output_delta(const Matrix& target, const Matrix& predicted,
                            const Matrix& preactivation,
                            double (*output_activation_df)(double)) const {
    if (output_activation_df == nullptr) {
        throw std::invalid_argument("MseLoss: output_activation_df cannot be null");
    }
    return (target - predicted).hadamard(preactivation.apply(output_activation_df));
}

double CrossEntropyLoss::compute(const Matrix& target, const Matrix& predicted) const {
    if (target.rows() == 0 || target.cols() == 0) {
        throw std::invalid_argument("CrossEntropyLoss: target has zero rows or columns");
    }
    if (target.rows() != predicted.rows() || target.cols() != predicted.cols()) {
        throw std::invalid_argument("CrossEntropyLoss: shape mismatch between target (" +
                                    std::to_string(target.rows()) + "x" + std::to_string(target.cols()) +
                                    ") and predicted (" + std::to_string(predicted.rows()) + "x" +
                                    std::to_string(predicted.cols()) + ")");
    }

    constexpr double kMinProb = 1e-12;
    constexpr double kMaxProb = 1.0 - 1e-12;

    double total_row_loss = 0.0;
    for (std::size_t i = 0; i < target.rows(); ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < target.cols(); ++j) {
            double p = std::clamp(predicted(i, j), kMinProb, kMaxProb);
            row_sum += target(i, j) * std::log(p);
        }
        total_row_loss += row_sum;
    }

    return -total_row_loss / static_cast<double>(target.rows());
}

Matrix CrossEntropyLoss::output_delta(const Matrix& target, const Matrix& predicted,
                                     const Matrix& preactivation,
                                     double (*output_activation_df)(double)) const {
    (void)preactivation;
    if (output_activation_df != nullptr) {
        throw std::invalid_argument(
            "CrossEntropyLoss: output_activation_df must be null (combined softmax/cross_entropy)");
    }
    if (target.rows() != predicted.rows() || target.cols() != predicted.cols()) {
        throw std::invalid_argument("CrossEntropyLoss::output_delta: shape mismatch");
    }
    return target - predicted;
}

const std::vector<std::string>& loss_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> v;
        for (const char* name : kLossNames) {
            v.emplace_back(name);
        }
        return v;
    }();
    return names;
}

std::unique_ptr<Loss> loss_by_name(const std::string& name) {
    if (name == "mse") {
        return std::make_unique<MseLoss>();
    }
    if (name == "cross_entropy") {
        return std::make_unique<CrossEntropyLoss>();
    }
    throw std::invalid_argument("unknown loss: '" + name + "' (expected one of: " + joined_loss_names() + ")");
}

}  // namespace tp3
