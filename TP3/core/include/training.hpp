#pragma once

#include <vector>

#include "matrix.hpp"

namespace tp3 {

// What `Model::fit` returns: the loss after each epoch, as data (never a plot).
struct TrainResult {
    std::vector<double> loss_per_epoch;
};

// Mean of (expected - predicted)^2 over all elements; shapes must match.
double mean_squared_error(const Matrix& expected, const Matrix& predicted);

}  // namespace tp3
