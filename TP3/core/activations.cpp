#include "activations.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tp3 {

namespace {

double step_f(double h) { return h >= 0.0 ? 1.0 : -1.0; }
// The perceptron rule: treating the step's derivative as 1 makes the shared
// update rule (delta = (y - o) * df(h)) reduce to the classic w += lr * (y - o) * x.
double step_df(double) { return 1.0; }

double identity_f(double h) { return h; }
double identity_df(double) { return 1.0; }

double tanh_f(double h) { return std::tanh(h); }
double tanh_df(double h) {
    const double t = std::tanh(h);
    return 1.0 - t * t;
}

double sigmoid_f(double h) { return 1.0 / (1.0 + std::exp(-h)); }
double sigmoid_df(double h) {
    const double s = sigmoid_f(h);
    return s * (1.0 - s);
}

const Activation kActivations[] = {
    {"step", step_f, step_df},
    {"identity", identity_f, identity_df},
    {"tanh", tanh_f, tanh_df},
    {"sigmoid", sigmoid_f, sigmoid_df},
};

std::string joined_names() {
    std::string out;
    for (const Activation& a : kActivations) {
        if (!out.empty()) {
            out += ", ";
        }
        out += a.name;
    }
    return out;
}

}  // namespace

const Activation& activation_by_name(const std::string& name) {
    for (const Activation& a : kActivations) {
        if (name.size() == std::strlen(a.name) && std::memcmp(name.data(), a.name, name.size()) == 0) {
            return a;
        }
    }
    throw std::invalid_argument("unknown activation: '" + name + "' (expected one of: " + joined_names() + ")");
}

const std::vector<std::string>& activation_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> v;
        for (const Activation& a : kActivations) {
            v.emplace_back(a.name);
        }
        return v;
    }();
    return names;
}

}  // namespace tp3
