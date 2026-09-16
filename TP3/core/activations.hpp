#pragma once

#include <string>
#include <vector>

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

}  // namespace tp3
