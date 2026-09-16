// tp3 — CLI for the TP3 neural-network engine.
//
//   tp3 validate <case> [--seed N] [--epochs N] [--lr F] [--out DIR]
//
// Runs validation case(s), prints the result and writes <DIR>/<case>.json.
// Exit 0 on success, 1 on runtime/engine errors, 2 on usage errors.

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "io/run_json.hpp"
#include "mlp.hpp"
#include "perceptron.hpp"
#include "validation/datasets.hpp"

namespace {

using tp3::Dataset;
using tp3::Matrix;

struct CaseSpec {
    const char* name;
    const char* activation;
    double learning_rate;
    int epochs;
    Dataset (*make_dataset)(std::mt19937_64&);
};

// Datasets draw from the run rng first (K-07); `and` and `xor` are fixed and draw nothing.
Dataset make_and(std::mt19937_64&) { return tp3::and_dataset(); }
Dataset make_linear(std::mt19937_64& rng) { return tp3::linear_dataset(50, rng); }
Dataset make_tanh(std::mt19937_64& rng) { return tp3::tanh_dataset(50, rng); }
Dataset make_xor(std::mt19937_64&) { return tp3::xor_dataset(); }

// lr / epochs are pinned per case (K-12); a run echoes them in its JSON hyperparameters.
const CaseSpec kCases[] = {
    {"and", "step", 0.1, 100, make_and},
    {"linear", "identity", 0.01, 200, make_linear},
    {"tanh", "tanh", 0.05, 500, make_tanh},
    {"xor", "tanh", 0.1, 2000, make_xor},
};

std::string case_names() {
    std::string out;
    for (const CaseSpec& c : kCases) {
        if (!out.empty()) {
            out += ", ";
        }
        out += c.name;
    }
    return out;
}

const CaseSpec* find_case(const std::string& name) {
    for (const CaseSpec& c : kCases) {
        if (name == c.name) {
            return &c;
        }
    }
    return nullptr;
}

void print_usage(std::ostream& os) {
    os << "usage: tp3 validate <case> [--seed N] [--epochs N] [--lr F] [--out DIR]\n"
       << "       tp3 --help\n"
       << "\n"
       << "cases: " << case_names() << "\n"
       << "defaults: --seed 42, --out runs_validation, --epochs/--lr per case\n";
}

int usage_error(const std::string& message) {
    std::cerr << message << "\n";
    print_usage(std::cerr);
    return 2;
}

bool parse_ull(const std::string& s, unsigned long long& out) {
    if (s.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0' || s[0] == '-') {
        return false;
    }
    out = v;
    return true;
}

bool parse_int(const std::string& s, int& out) {
    if (s.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0' || v > 2147483647L || v < -2147483647L) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

bool parse_double(const std::string& s, double& out) {
    if (s.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s.c_str(), &end);
    if (errno != 0 || end == s.c_str() || *end != '\0') {
        return false;
    }
    out = v;
    return true;
}

void report_and_save(const std::string& json_case_name,
                     const std::string& display_title,
                     tp3::Model& model,
                     const Dataset& data,
                     const std::string& activation_name,
                     double learning_rate,
                     int epochs,
                     unsigned long long seed,
                     const std::vector<std::size_t>& layer_sizes,
                     double bias_scalar,
                     bool is_classification,
                     bool is_step,
                     const std::filesystem::path& out_dir) {
    const tp3::TrainResult train = model.fit(data.X, data.y, epochs);
    const Matrix predicted = model.predict(data.X);
    const std::vector<double> weights = model.flat_weights();
    const std::size_t n_inputs = data.X.cols();

    double final_bias = bias_scalar;
    if (auto* sp = dynamic_cast<tp3::SimplePerceptron*>(&model)) {
        final_bias = sp->bias();
    }

    std::cout << "\n=== " << display_title << " ===\n";
    std::cout << "case=" << json_case_name << " seed=" << seed << " activation=" << activation_name
              << " lr=" << learning_rate << " epochs=" << epochs << "\n";
    std::cout << "final weights: [";
    for (std::size_t j = 0; j < weights.size(); ++j) {
        std::cout << (j > 0 ? ", " : "") << weights[j];
    }
    std::cout << "]\n";
    if (final_bias != 0.0 || layer_sizes.empty()) {
        std::cout << "bias: " << final_bias << "\n";
    }
    std::cout << "final error (mse): " << train.loss_per_epoch.back() << "\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(24) << std::left << "input" << std::setw(12) << "expected" << "predicted\n";
    std::size_t correct = 0;
    tp3::RunRecord record;
    for (std::size_t i = 0; i < data.X.rows(); ++i) {
        tp3::PredictionRecord p;
        std::string input_text = "[";
        for (std::size_t j = 0; j < n_inputs; ++j) {
            p.input.push_back(data.X(i, j));
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s%.4f", j > 0 ? ", " : "", data.X(i, j));
            input_text += buf;
        }
        input_text += "]";
        p.expected = data.y(i, 0);
        p.predicted = predicted(i, 0);
        if (is_classification) {
            const double class_pred = is_step ? p.predicted : (p.predicted >= 0.0 ? 1.0 : -1.0);
            if (class_pred == p.expected) {
                ++correct;
            }
        }
        std::cout << std::setw(24) << std::left << input_text << std::setw(12) << p.expected << p.predicted
                  << "\n";
        record.predictions.push_back(p);
    }
    std::cout.unsetf(std::ios::fixed);
    std::cout << std::setprecision(6);
    if (is_classification) {
        std::cout << "correct: " << correct << "/" << data.X.rows();
        if (is_step && correct < data.X.rows()) {
            std::cout << " (step perceptron cannot solve non-linearly separable XOR)";
        }
        std::cout << "\n";
    }

    record.case_name = json_case_name;
    record.seed = seed;
    record.activation = activation_name;
    record.learning_rate = learning_rate;
    record.epochs = epochs;
    record.n_inputs = n_inputs;
    record.layer_sizes = layer_sizes;
    record.loss_per_epoch = train.loss_per_epoch;
    record.final_weights = weights;
    record.bias = final_bias;

    const std::filesystem::path written = tp3::write_run_json(record, out_dir);
    std::cout << "wrote: " << written.string() << "\n";

    // For compatibility with single-case runners expecting xor.json
    if (json_case_name == "xor_221") {
        tp3::RunRecord default_xor = record;
        default_xor.case_name = "xor";
        tp3::write_run_json(default_xor, out_dir);
    }
}

int run_validate(int argc, char** argv) {
    if (argc < 3) {
        return usage_error("missing case name");
    }
    const std::string case_name = argv[2];
    const CaseSpec* spec = find_case(case_name);
    if (spec == nullptr) {
        return usage_error("unknown case: " + case_name + " (expected one of: " + case_names() + ")");
    }

    unsigned long long seed = 42;
    int custom_epochs = -1;
    double custom_lr = -1.0;
    std::filesystem::path out_dir = "runs_validation";

    for (int i = 3; i < argc; ++i) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) {
            return usage_error("flag " + flag + " needs a value");
        }
        const std::string value = argv[++i];
        if (flag == "--seed") {
            if (!parse_ull(value, seed)) {
                return usage_error("invalid --seed: " + value);
            }
        } else if (flag == "--epochs") {
            int ep = 0;
            if (!parse_int(value, ep) || ep <= 0) {
                return usage_error("invalid --epochs (must be a positive integer): " + value);
            }
            custom_epochs = ep;
        } else if (flag == "--lr") {
            double lr = 0.0;
            if (!parse_double(value, lr) || !(lr > 0.0) || !std::isfinite(lr)) {
                return usage_error("invalid --lr (must be a positive finite number): " + value);
            }
            custom_lr = lr;
        } else if (flag == "--out") {
            out_dir = value;
        } else {
            return usage_error("unknown flag: " + flag);
        }
    }

    if (case_name == "xor") {
        const Dataset data = tp3::xor_dataset();
        const double lr_mlp = custom_lr > 0.0 ? custom_lr : 0.1;
        const int epochs_mlp = custom_epochs > 0 ? custom_epochs : 2000;
        const double lr_step = custom_lr > 0.0 ? custom_lr : 0.1;
        const int epochs_step = custom_epochs > 0 ? custom_epochs : 100;

        // Run 1: MLP [2, 2, 1]
        std::mt19937_64 rng_221(seed);
        tp3::MLP mlp_221({2, 2, 1}, "tanh", lr_mlp, rng_221);
        report_and_save("xor_221", "XOR — MLP [2, 2, 1]", mlp_221, data, "tanh", lr_mlp, epochs_mlp,
                        seed, {2, 2, 1}, 0.0, true, false, out_dir);

        // Run 2: MLP [2, 3, 2, 1]
        std::mt19937_64 rng_2321(seed);
        tp3::MLP mlp_2321({2, 3, 2, 1}, "tanh", lr_mlp, rng_2321);
        report_and_save("xor_2321", "XOR — MLP [2, 3, 2, 1]", mlp_2321, data, "tanh", lr_mlp,
                        epochs_mlp, seed, {2, 3, 2, 1}, 0.0, true, false, out_dir);

        // Run 3: Step Perceptron (demonstrates failure on XOR)
        std::mt19937_64 rng_step(seed);
        tp3::SimplePerceptron step_model(2, "step", lr_step, rng_step);
        report_and_save("xor_step", "XOR — Step Perceptron (fails)", step_model, data, "step",
                        lr_step, epochs_step, seed, {}, step_model.bias(), true, true, out_dir);

        return 0;
    }

    // Standard cases: and, linear, tanh
    std::mt19937_64 rng(seed);
    const Dataset data = spec->make_dataset(rng);
    const std::size_t n_inputs = data.X.cols();
    const double lr = custom_lr > 0.0 ? custom_lr : spec->learning_rate;
    const int epochs = custom_epochs > 0 ? custom_epochs : spec->epochs;
    const bool is_step = std::strcmp(spec->activation, "step") == 0;

    tp3::SimplePerceptron model(n_inputs, spec->activation, lr, rng);
    report_and_save(spec->name, spec->name, model, data, spec->activation, lr, epochs, seed, {},
                    model.bias(), is_step, is_step, out_dir);

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        return usage_error("missing subcommand");
    }
    const std::string command = argv[1];
    if (command == "--help" || command == "-h" || command == "help") {
        print_usage(std::cout);
        return 0;
    }
    if (command == "validate") {
        try {
            return run_validate(argc, argv);
        } catch (const std::exception& e) {
            std::cerr << "error: " << e.what() << "\n";
            return 1;
        }
    }
    return usage_error("unknown subcommand: " + command);
}
