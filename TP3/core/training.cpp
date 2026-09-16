#include "training.hpp"

#include <stdexcept>
#include <string>

namespace tp3 {

double mean_squared_error(const Matrix& expected, const Matrix& predicted) {
    if (expected.rows() != predicted.rows() || expected.cols() != predicted.cols()) {
        throw std::invalid_argument("mean_squared_error: " + std::to_string(expected.rows()) + "x" +
                                    std::to_string(expected.cols()) + " vs " +
                                    std::to_string(predicted.rows()) + "x" +
                                    std::to_string(predicted.cols()));
    }
    const std::vector<double>& e = expected.data();
    const std::vector<double>& p = predicted.data();
    if (e.empty()) {
        throw std::invalid_argument("mean_squared_error: empty matrices");
    }
    double sum = 0.0;
    for (std::size_t k = 0; k < e.size(); ++k) {
        const double d = e[k] - p[k];
        sum += d * d;
    }
    return sum / static_cast<double>(e.size());
}

}  // namespace tp3
