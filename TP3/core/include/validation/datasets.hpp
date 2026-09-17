#pragma once

#include <cstddef>
#include <random>

#include "matrix.hpp"

namespace tp3 {

// Inputs X (N x n_inputs) and targets y (N x 1) for one validation case.
struct Dataset {
    Matrix X;
    Matrix y;
};

// AND with bipolar inputs, in this fixed order:
// X = {{-1,1},{1,-1},{-1,-1},{1,1}}, y = {-1,-1,-1,1}.
Dataset and_dataset();

// n samples of y = x: x_i drawn in index order from U(-2, 2) on rng (K-11).
// X is n x 1, y is n x 1, no noise.
Dataset linear_dataset(std::size_t n, std::mt19937_64& rng);

// n samples of y = tanh(x): same sampler as linear_dataset (K-11).
Dataset tanh_dataset(std::size_t n, std::mt19937_64& rng);

// XOR with bipolar inputs, in this fixed order:
// X = {{-1,1},{1,-1},{-1,-1},{1,1}}, y = {1,1,-1,-1}.
Dataset xor_dataset();

}  // namespace tp3
