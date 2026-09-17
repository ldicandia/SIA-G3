#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "matrix.hpp"

namespace tp3 {

// Abstract base class for optimization algorithms.
class Optimizer {
public:
    virtual ~Optimizer() = default;

    // Updates the parameter matrix in-place using the given gradient and distinct parameter ID.
    virtual void update(Matrix& param, const Matrix& grad, std::size_t param_id) = 0;

    virtual const char* name() const = 0;
};

class SgdOptimizer : public Optimizer {
public:
    explicit SgdOptimizer(double learning_rate);

    void update(Matrix& param, const Matrix& grad, std::size_t param_id) override;
    const char* name() const override { return "sgd"; }
    double learning_rate() const { return learning_rate_; }

private:
    double learning_rate_;
};

class MomentumOptimizer : public Optimizer {
public:
    explicit MomentumOptimizer(double learning_rate, double momentum = 0.9);

    void update(Matrix& param, const Matrix& grad, std::size_t param_id) override;
    const char* name() const override { return "momentum"; }
    double learning_rate() const { return learning_rate_; }
    double momentum() const { return momentum_; }

private:
    double learning_rate_;
    double momentum_;
    std::unordered_map<std::size_t, Matrix> velocity_;
};

class AdamOptimizer : public Optimizer {
public:
    explicit AdamOptimizer(double learning_rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

    void update(Matrix& param, const Matrix& grad, std::size_t param_id) override;
    const char* name() const override { return "adam"; }
    double learning_rate() const { return learning_rate_; }
    double beta1() const { return beta1_; }
    double beta2() const { return beta2_; }
    double epsilon() const { return epsilon_; }

private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    std::unordered_map<std::size_t, Matrix> m_;
    std::unordered_map<std::size_t, Matrix> v_;
    std::unordered_map<std::size_t, int> t_;
};

const std::vector<std::string>& optimizer_names();

std::unique_ptr<Optimizer> optimizer_by_name(const std::string& name, double learning_rate,
                                             double momentum_coefficient = 0.9, double beta1 = 0.9,
                                             double beta2 = 0.999, double epsilon = 1e-8);

}  // namespace tp3
