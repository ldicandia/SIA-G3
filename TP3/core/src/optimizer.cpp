#include "optimizer.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tp3 {

namespace {

const char* const kOptimizerNames[] = {
    "sgd",
    "momentum",
    "adam",
};

std::string joined_optimizer_names() {
    std::string out;
    for (const char* name : kOptimizerNames) {
        if (!out.empty()) {
            out += ", ";
        }
        out += name;
    }
    return out;
}

}  // namespace

SgdOptimizer::SgdOptimizer(double learning_rate)
    : learning_rate_(learning_rate) {
    if (!std::isfinite(learning_rate) || !(learning_rate > 0.0)) {
        throw std::invalid_argument("SgdOptimizer: learning_rate must be positive and finite");
    }
}

void SgdOptimizer::update(Matrix& param, const Matrix& grad, std::size_t /*param_id*/) {
    if (param.rows() != grad.rows() || param.cols() != grad.cols()) {
        throw std::invalid_argument("SgdOptimizer::update: param and grad shapes mismatch");
    }
    param = param + (grad * learning_rate_);
}

MomentumOptimizer::MomentumOptimizer(double learning_rate, double momentum)
    : learning_rate_(learning_rate), momentum_(momentum) {
    if (!std::isfinite(learning_rate) || !(learning_rate > 0.0)) {
        throw std::invalid_argument("MomentumOptimizer: learning_rate must be positive and finite");
    }
    if (!std::isfinite(momentum) || momentum < 0.0 || momentum >= 1.0) {
        throw std::invalid_argument("MomentumOptimizer: momentum must be in [0.0, 1.0)");
    }
}

void MomentumOptimizer::update(Matrix& param, const Matrix& grad, std::size_t param_id) {
    if (param.rows() != grad.rows() || param.cols() != grad.cols()) {
        throw std::invalid_argument("MomentumOptimizer::update: param and grad shapes mismatch");
    }
    auto it = velocity_.find(param_id);
    if (it == velocity_.end() || it->second.rows() != grad.rows() || it->second.cols() != grad.cols()) {
        it = velocity_.insert_or_assign(param_id, Matrix(grad.rows(), grad.cols(), 0.0)).first;
    }

    it->second = (it->second * momentum_) + (grad * learning_rate_);
    param = param + it->second;
}

AdamOptimizer::AdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
    if (!std::isfinite(learning_rate) || !(learning_rate > 0.0)) {
        throw std::invalid_argument("AdamOptimizer: learning_rate must be positive and finite");
    }
    if (!std::isfinite(beta1) || beta1 < 0.0 || beta1 >= 1.0) {
        throw std::invalid_argument("AdamOptimizer: beta1 must be in [0.0, 1.0)");
    }
    if (!std::isfinite(beta2) || beta2 < 0.0 || beta2 >= 1.0) {
        throw std::invalid_argument("AdamOptimizer: beta2 must be in [0.0, 1.0)");
    }
    if (!std::isfinite(epsilon) || !(epsilon > 0.0)) {
        throw std::invalid_argument("AdamOptimizer: epsilon must be positive and finite");
    }
}

void AdamOptimizer::update(Matrix& param, const Matrix& grad, std::size_t param_id) {
    if (param.rows() != grad.rows() || param.cols() != grad.cols()) {
        throw std::invalid_argument("AdamOptimizer::update: param and grad shapes mismatch");
    }

    auto it_m = m_.find(param_id);
    if (it_m == m_.end() || it_m->second.rows() != grad.rows() || it_m->second.cols() != grad.cols()) {
        m_.insert_or_assign(param_id, Matrix(grad.rows(), grad.cols(), 0.0));
        v_.insert_or_assign(param_id, Matrix(grad.rows(), grad.cols(), 0.0));
        t_.insert_or_assign(param_id, 0);
    }

    t_[param_id] += 1;
    const int t = t_[param_id];

    Matrix& m = m_[param_id];
    Matrix& v = v_[param_id];

    m = (m * beta1_) + (grad * (1.0 - beta1_));
    Matrix grad_sq(grad.rows(), grad.cols());
    for (std::size_t r = 0; r < grad.rows(); ++r) {
        for (std::size_t c = 0; c < grad.cols(); ++c) {
            double g = grad(r, c);
            grad_sq(r, c) = g * g;
        }
    }
    v = (v * beta2_) + (grad_sq * (1.0 - beta2_));

    const double bias_correction1 = 1.0 - std::pow(beta1_, t);
    const double bias_correction2 = 1.0 - std::pow(beta2_, t);

    for (std::size_t r = 0; r < param.rows(); ++r) {
        for (std::size_t c = 0; c < param.cols(); ++c) {
            double m_hat = m(r, c) / bias_correction1;
            double v_hat = v(r, c) / bias_correction2;
            param(r, c) += learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}

const std::vector<std::string>& optimizer_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> vec;
        for (const char* name : kOptimizerNames) {
            vec.emplace_back(name);
        }
        return vec;
    }();
    return names;
}

std::unique_ptr<Optimizer> optimizer_by_name(const std::string& name, double learning_rate,
                                             double momentum_coefficient, double beta1,
                                             double beta2, double epsilon) {
    if (name == "sgd") {
        return std::make_unique<SgdOptimizer>(learning_rate);
    }
    if (name == "momentum") {
        return std::make_unique<MomentumOptimizer>(learning_rate, momentum_coefficient);
    }
    if (name == "adam") {
        return std::make_unique<AdamOptimizer>(learning_rate, beta1, beta2, epsilon);
    }
    throw std::invalid_argument("unknown optimizer: '" + name + "' (expected one of: " + joined_optimizer_names() + ")");
}

}  // namespace tp3
