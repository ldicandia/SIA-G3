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

#include <chrono>
#include <fstream>
#include <sstream>

#include "config.hpp"
#include "data/csv.hpp"
#include "io/model_io.hpp"
#include "io/run_json.hpp"
#include "loss.hpp"
#include "mlp.hpp"
#include "optimizer.hpp"
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
       << "       tp3 train --config <path>.json [--out DIR] [--save-model PATH] [--resume-from PATH]\n"
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

int run_train(int argc, char** argv) {
    std::string config_path;
    std::string out_dir_override;
    std::string save_model_path;
    std::string resume_from_path;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config") {
            if (i + 1 >= argc) {
                return usage_error("missing argument for --config");
            }
            config_path = argv[++i];
        } else if (arg == "--out") {
            if (i + 1 >= argc) {
                return usage_error("missing argument for --out");
            }
            out_dir_override = argv[++i];
        } else if (arg == "--save-model") {
            if (i + 1 >= argc) {
                return usage_error("missing argument for --save-model");
            }
            save_model_path = argv[++i];
        } else if (arg == "--resume-from") {
            if (i + 1 >= argc) {
                return usage_error("missing argument for --resume-from");
            }
            resume_from_path = argv[++i];
        } else {
            return usage_error("unknown flag for train: " + arg);
        }
    }

    if (config_path.empty()) {
        return usage_error("missing required --config flag");
    }

    std::ifstream f(config_path);
    if (!f.is_open()) {
        std::cerr << "error: cannot open config file: " << config_path << "\n";
        return 2;
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string json_text = buffer.str();
    if (json_text.empty()) {
        std::cerr << "error: config file is empty: " << config_path << "\n";
        return 2;
    }

    tp3::RunConfig config;
    try {
        config = tp3::RunConfig::parse(json_text);
    } catch (const std::exception& e) {
        std::string msg = e.what();
        std::cerr << "error: " << msg << "\n";
        if (msg.rfind("unknown ", 0) == 0) {
            return 1;
        }
        return 2;
    }

    if (!out_dir_override.empty()) {
        config.output_dir = out_dir_override;
    }
    if (!resume_from_path.empty()) {
        config.resume_from = resume_from_path;
    }

    tp3::ModelSnapshot snap;
    bool is_resuming = !config.resume_from.empty();
    if (is_resuming) {
        try {
            snap = tp3::load_model(config.resume_from);
        } catch (const std::exception& e) {
            std::cerr << "error loading model to resume: " << e.what() << "\n";
            return 2;
        }

        if (config.model_type != snap.model_type) {
            std::cerr << "error: model_type mismatch: config has '" << config.model_type
                      << "', snapshot has '" << snap.model_type << "'\n";
            return 2;
        }
        if (config.model_type == "mlp" && config.layer_sizes != snap.layer_sizes) {
            std::cerr << "error: layer_sizes mismatch between config and snapshot\n";
            return 2;
        }
        if (config.activation != snap.activation) {
            std::cerr << "error: activation mismatch: config has '" << config.activation
                      << "', snapshot has '" << snap.activation << "'\n";
            return 2;
        }
        if (config.use_softmax_output != snap.use_softmax_output) {
            std::cerr << "error: use_softmax_output mismatch between config and snapshot\n";
            return 2;
        }
        if (config.loss != snap.run_config.loss) {
            std::cerr << "error: loss mismatch: config has '" << config.loss
                      << "', snapshot has '" << snap.run_config.loss << "'\n";
            return 2;
        }
        if (config.optimizer_name != snap.run_config.optimizer_name) {
            std::cerr << "error: optimizer mismatch: config has '" << config.optimizer_name
                      << "', snapshot has '" << snap.run_config.optimizer_name << "'\n";
            return 2;
        }
        if (snap.epochs_completed > config.epochs) {
            std::cerr << "error: snapshot already completed " << snap.epochs_completed
                      << " epochs; config requested " << config.epochs << "\n";
            return 2;
        }
    }

    std::mt19937_64 rng(config.seed);
    Dataset data;
    if (config.dataset_kind == "validation") {
        if (config.dataset_case == "and") {
            data = tp3::and_dataset();
        } else if (config.dataset_case == "linear") {
            data = tp3::linear_dataset(50, rng);
        } else if (config.dataset_case == "tanh") {
            data = tp3::tanh_dataset(50, rng);
        } else if (config.dataset_case == "xor") {
            data = tp3::xor_dataset();
        } else {
            std::cerr << "error: unknown dataset_case: " << config.dataset_case << "\n";
            return 2;
        }
    } else if (config.dataset_kind == "csv") {
        if (config.dataset_format == "plain") {
            auto plain = tp3::load_plain_csv(config.dataset_path, config.dataset_target_column);
            data.X = std::move(plain.X);
            data.y = std::move(plain.y);
        } else if (config.dataset_format == "digits") {
            auto img_csv = tp3::load_labeled_image_csv(config.dataset_path);
            data.X = std::move(img_csv.images);
            std::size_t num_classes = config.layer_sizes.back();
            Matrix y_onehot(img_csv.labels.rows(), num_classes, 0.0);
            for (std::size_t r = 0; r < img_csv.labels.rows(); ++r) {
                std::size_t lbl = static_cast<std::size_t>(img_csv.labels(r, 0));
                if (lbl < num_classes) {
                    y_onehot(r, lbl) = 1.0;
                }
            }
            data.y = std::move(y_onehot);
        } else {
            std::cerr << "error: unknown dataset_format: " << config.dataset_format << "\n";
            return 2;
        }
    } else {
        std::cerr << "error: unknown dataset_kind: " << config.dataset_kind << "\n";
        return 2;
    }

    std::unique_ptr<tp3::Model> model;
    if (config.model_type == "mlp") {
        auto loss = tp3::loss_by_name(config.loss);
        auto opt = tp3::optimizer_by_name(config.optimizer_name, config.learning_rate,
                                          config.momentum_coefficient, config.adam_beta1,
                                          config.adam_beta2, config.adam_epsilon);
        model = std::make_unique<tp3::MLP>(config.layer_sizes, config.activation,
                                            config.learning_rate, rng,
                                            std::move(loss), std::move(opt),
                                            config.use_softmax_output);
    } else if (config.model_type == "perceptron") {
        auto loss = tp3::loss_by_name(config.loss);
        auto opt = tp3::optimizer_by_name(config.optimizer_name, config.learning_rate,
                                          config.momentum_coefficient, config.adam_beta1,
                                          config.adam_beta2, config.adam_epsilon);
        model = std::make_unique<tp3::SimplePerceptron>(data.X.cols(), config.activation,
                                                        config.learning_rate, rng,
                                                        std::move(loss), std::move(opt));
    } else {
        std::cerr << "error: unknown model_type: " << config.model_type << "\n";
        return 1;
    }

    int epochs_to_train = config.epochs;
    int start_epoch = 0;
    if (is_resuming) {
        start_epoch = snap.epochs_completed;
        epochs_to_train = config.epochs - snap.epochs_completed;
        std::vector<Matrix> w_in, b_in;
        tp3::unflatten_into(snap.flat_weights, snap.layer_sizes, w_in, b_in);
        if (config.model_type == "mlp") {
            static_cast<tp3::MLP*>(model.get())->set_weights_and_biases(w_in, b_in);
        } else {
            static_cast<tp3::SimplePerceptron*>(model.get())->set_weights(w_in[0], b_in[0](0, 0));
        }
    }

    auto progress_cb = [&](int epoch, double loss) {
        int global_epoch = start_epoch + epoch;
        if ((global_epoch % config.progress_interval_epochs == 0) || (global_epoch == config.epochs)) {
            std::cerr << "epoch=" << global_epoch << " loss=" << loss << "\n";
        }
    };

    tp3::TrainResult train_result;
    auto start_time = std::chrono::steady_clock::now();
    if (epochs_to_train > 0) {
        train_result = model->fit(data.X, data.y, epochs_to_train, progress_cb);
    }
    auto end_time = std::chrono::steady_clock::now();
    double wall_time = std::chrono::duration<double>(end_time - start_time).count();

    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::string prefix = config.dataset_case.empty() ? config.model_type : config.dataset_case;
    std::string case_id = prefix + "_" + std::to_string(config.seed) + "_" + std::to_string(now_us);

    tp3::RunRecord record;
    record.case_name = case_id;
    record.seed = config.seed;
    record.activation = config.activation;
    record.learning_rate = config.learning_rate;
    record.epochs = config.epochs;
    record.n_inputs = data.X.cols();
    record.layer_sizes = config.layer_sizes;

    std::vector<double> full_loss;
    if (is_resuming) {
        full_loss = snap.loss_per_epoch_so_far;
    }
    full_loss.insert(full_loss.end(), train_result.loss_per_epoch.begin(), train_result.loss_per_epoch.end());
    record.loss_per_epoch = std::move(full_loss);

    record.final_weights = model->flat_weights();
    record.bias = 0.0;
    if (auto* sp = dynamic_cast<tp3::SimplePerceptron*>(model.get())) {
        record.bias = sp->bias();
    }
    record.loss_name = config.loss;
    record.optimizer_name = config.optimizer_name;
    record.dataset_path = config.dataset_path;
    record.wall_time_seconds = wall_time;

    Matrix preds = model->predict(data.X);
    for (std::size_t i = 0; i < data.X.rows(); ++i) {
        tp3::PredictionRecord pr;
        for (std::size_t j = 0; j < data.X.cols(); ++j) {
            pr.input.push_back(data.X(i, j));
        }
        pr.expected = data.y(i, 0);
        pr.predicted = preds(i, 0);
        record.predictions.push_back(pr);
    }

    std::filesystem::path written = tp3::write_run_json(record, config.output_dir);
    std::cout << "run metrics written to " << written.string() << "\n";

    if (!save_model_path.empty()) {
        tp3::ModelSnapshot snap_to_save;
        snap_to_save.model_type = config.model_type;
        snap_to_save.activation = config.activation;
        snap_to_save.use_softmax_output = config.use_softmax_output;
        snap_to_save.epochs_completed = config.epochs;
        snap_to_save.loss_per_epoch_so_far = record.loss_per_epoch;
        snap_to_save.run_config = config;

        if (config.model_type == "mlp") {
            auto* mlp = static_cast<tp3::MLP*>(model.get());
            snap_to_save.layer_sizes = config.layer_sizes;
            for (std::size_t l = 0; l < mlp->weights().size(); ++l) {
                for (double w : mlp->weights()[l].data()) snap_to_save.flat_weights.push_back(w);
                for (double b : mlp->biases()[l].data()) snap_to_save.flat_weights.push_back(b);
            }
        } else {
            auto* sp = static_cast<tp3::SimplePerceptron*>(model.get());
            snap_to_save.layer_sizes = {sp->n_inputs(), 1};
            for (double w : sp->weights().data()) snap_to_save.flat_weights.push_back(w);
            snap_to_save.flat_weights.push_back(sp->bias());
        }
        tp3::save_model(snap_to_save, save_model_path);
    }

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
    if (command == "train") {
        try {
            return run_train(argc, argv);
        } catch (const std::exception& e) {
            std::cerr << "error: " << e.what() << "\n";
            return 1;
        }
    }
    return usage_error("unknown subcommand: " + command);
}
