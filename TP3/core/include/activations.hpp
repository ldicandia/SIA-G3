#pragma once

#include <string>
#include <vector>

#include "matrix.hpp"

namespace tp3 {

// An activation paired with its derivative. `df` takes the pre-activation h.
struct Activation {
    const char* name;
    double (*f)(double);
    double (*df)(double);
};

// Byte-exact, case-sensitive lookup; throws std::invalid_argument on unknown or empty name.
const Activation& activation_by_name(const std::string& name);

// Fixed order: step, identity, tanh, sigmoid.
const std::vector<std::string>& activation_names();

// Row-wise softmax with numerical stability (subtracts max logit per row).
// Throws std::invalid_argument if logits has zero rows or zero columns.
Matrix softmax_rows(const Matrix& logits);

}  // namespace tp3
