#include "config.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "activations.hpp"
#include "io/json_value.hpp"
#include "loss.hpp"
#include "optimizer.hpp"

namespace tp3 {

namespace {

bool contains(const std::vector<std::string>& vec, const std::string& val) {
    for (const auto& item : vec) {
        if (item == val) return true;
    }
    return false;
}

std::string join(const std::vector<std::string>& vec) {
    std::string out;
    for (const auto& item : vec) {
        if (!out.empty()) out += ", ";
        out += item;
    }
    return out;
}

}  // namespace

RunConfig RunConfig::parse(const std::string& json_text) {
    JsonObject obj = JsonObject::parse(json_text);
    RunConfig cfg;

    if (obj.has("format_version")) {
        cfg.format_version = static_cast<int>(obj.get_int("format_version"));
        if (cfg.format_version != 1) {
            throw std::invalid_argument("unsupported format_version: " + std::to_string(cfg.format_version) +
                                        " (expected 1)");
        }
    }

    // Required keys
    const std::vector<std::string> required_keys = {
        "model_type", "learning_rate", "epochs", "seed", "loss", "optimizer", "dataset_kind"
    };
    for (const auto& key : required_keys) {
        if (!obj.has(key)) {
            throw std::invalid_argument("missing required configuration key: '" + key + "'");
        }
    }

    cfg.model_type = obj.get_string("model_type");
    const std::vector<std::string> allowed_models = {"mlp", "perceptron"};
    if (!contains(allowed_models, cfg.model_type)) {
        throw std::invalid_argument("unknown model_type: '" + cfg.model_type +
                                    "' (expected one of: " + join(allowed_models) + ")");
    }

    if (!obj.has("activation")) {
        throw std::invalid_argument("missing required configuration key: 'activation'");
    }
    cfg.activation = obj.get_string("activation");
    if (!contains(activation_names(), cfg.activation)) {
        throw std::invalid_argument("unknown activation: '" + cfg.activation +
                                    "' (expected one of: " + join(activation_names()) + ")");
    }

    cfg.loss = obj.get_string("loss");
    if (!contains(loss_names(), cfg.loss)) {
        throw std::invalid_argument("unknown loss: '" + cfg.loss +
                                    "' (expected one of: " + join(loss_names()) + ")");
    }

    cfg.optimizer_name = obj.get_string("optimizer");
    if (!contains(optimizer_names(), cfg.optimizer_name)) {
        throw std::invalid_argument("unknown optimizer: '" + cfg.optimizer_name +
                                    "' (expected one of: " + join(optimizer_names()) + ")");
    }

    cfg.learning_rate = obj.get_number("learning_rate");
    if (!std::isfinite(cfg.learning_rate) || cfg.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be a positive finite number, got " +
                                    std::to_string(cfg.learning_rate));
    }

    cfg.epochs = static_cast<int>(obj.get_int("epochs"));
    if (cfg.epochs <= 0) {
        throw std::invalid_argument("epochs must be > 0, got " + std::to_string(cfg.epochs));
    }

    cfg.seed = static_cast<unsigned long long>(obj.get_int("seed"));

    cfg.dataset_kind = obj.get_string("dataset_kind");
    const std::vector<std::string> allowed_datasets = {"validation", "csv"};
    if (!contains(allowed_datasets, cfg.dataset_kind)) {
        throw std::invalid_argument("unknown dataset_kind: '" + cfg.dataset_kind +
                                    "' (expected one of: " + join(allowed_datasets) + ")");
    }

    if (cfg.dataset_kind == "validation") {
        if (!obj.has("dataset_case")) {
            throw std::invalid_argument("missing required key 'dataset_case' for validation dataset");
        }
        cfg.dataset_case = obj.get_string("dataset_case");
        const std::vector<std::string> allowed_cases = {"and", "linear", "tanh", "xor"};
        if (!contains(allowed_cases, cfg.dataset_case)) {
            throw std::invalid_argument("unknown dataset_case: '" + cfg.dataset_case +
                                        "' (expected one of: " + join(allowed_cases) + ")");
        }
    } else if (cfg.dataset_kind == "csv") {
        if (!obj.has("dataset_path")) {
            throw std::invalid_argument("missing required key 'dataset_path' for csv dataset");
        }
        cfg.dataset_path = obj.get_string("dataset_path");

        if (!obj.has("dataset_format")) {
            throw std::invalid_argument("missing required key 'dataset_format' for csv dataset");
        }
        cfg.dataset_format = obj.get_string("dataset_format");
        const std::vector<std::string> allowed_formats = {"plain", "digits"};
        if (!contains(allowed_formats, cfg.dataset_format)) {
            throw std::invalid_argument("unknown dataset_format: '" + cfg.dataset_format +
                                        "' (expected one of: " + join(allowed_formats) + ")");
        }

        if (cfg.dataset_format == "plain") {
            if (!obj.has("dataset_target_column")) {
                throw std::invalid_argument("missing required key 'dataset_target_column' for plain csv dataset");
            }
            cfg.dataset_target_column = obj.get_string("dataset_target_column");
        }
    }

    if (cfg.model_type == "mlp") {
        if (!obj.has("layer_sizes")) {
            throw std::invalid_argument("missing required key 'layer_sizes' for mlp model");
        }
        std::vector<double> sizes = obj.get_number_array("layer_sizes");
        if (sizes.size() < 2) {
            throw std::invalid_argument("layer_sizes must have at least 2 layers");
        }
        cfg.layer_sizes.clear();
        for (double s : sizes) {
            if (s <= 0.0 || std::floor(s) != s) {
                throw std::invalid_argument("layer_sizes must contain positive integers");
            }
            cfg.layer_sizes.push_back(static_cast<std::size_t>(s));
        }
    }

    // Optional keys
    cfg.use_softmax_output = obj.get_bool_or("use_softmax_output", false);
    cfg.momentum_coefficient = obj.get_number_or("momentum_coefficient", 0.9);
    cfg.adam_beta1 = obj.get_number_or("adam_beta1", 0.9);
    cfg.adam_beta2 = obj.get_number_or("adam_beta2", 0.999);
    cfg.adam_epsilon = obj.get_number_or("adam_epsilon", 1e-8);
    cfg.output_dir = obj.get_string_or("output_dir", "runs");
    cfg.progress_interval_epochs = static_cast<int>(obj.get_int_or("progress_interval_epochs", 50));
    cfg.resume_from = obj.get_string_or("resume_from", "");

    return cfg;
}

std::string RunConfig::to_json() const {
    std::vector<std::pair<std::string, JsonField>> fields;
    fields.emplace_back("format_version", JsonField::make_number(format_version));
    fields.emplace_back("model_type", JsonField::make_string(model_type));
    if (!layer_sizes.empty()) {
        std::vector<double> sizes;
        for (auto s : layer_sizes) sizes.push_back(static_cast<double>(s));
        fields.emplace_back("layer_sizes", JsonField::make_number_array(sizes));
    }
    fields.emplace_back("activation", JsonField::make_string(activation));
    fields.emplace_back("use_softmax_output", JsonField::make_bool(use_softmax_output));
    fields.emplace_back("loss", JsonField::make_string(loss));
    fields.emplace_back("optimizer", JsonField::make_string(optimizer_name));
    fields.emplace_back("momentum_coefficient", JsonField::make_number(momentum_coefficient));
    fields.emplace_back("adam_beta1", JsonField::make_number(adam_beta1));
    fields.emplace_back("adam_beta2", JsonField::make_number(adam_beta2));
    fields.emplace_back("adam_epsilon", JsonField::make_number(adam_epsilon));
    fields.emplace_back("learning_rate", JsonField::make_number(learning_rate));
    fields.emplace_back("epochs", JsonField::make_number(epochs));
    fields.emplace_back("seed", JsonField::make_number(static_cast<double>(seed)));
    fields.emplace_back("dataset_kind", JsonField::make_string(dataset_kind));
    if (!dataset_case.empty()) {
        fields.emplace_back("dataset_case", JsonField::make_string(dataset_case));
    }
    if (!dataset_format.empty()) {
        fields.emplace_back("dataset_format", JsonField::make_string(dataset_format));
    }
    if (!dataset_path.empty()) {
        fields.emplace_back("dataset_path", JsonField::make_string(dataset_path));
    }
    if (!dataset_target_column.empty()) {
        fields.emplace_back("dataset_target_column", JsonField::make_string(dataset_target_column));
    }
    fields.emplace_back("output_dir", JsonField::make_string(output_dir));
    fields.emplace_back("progress_interval_epochs", JsonField::make_number(progress_interval_epochs));
    if (!resume_from.empty()) {
        fields.emplace_back("resume_from", JsonField::make_string(resume_from));
    }

    return json_object_to_string(fields);
}

}  // namespace tp3
