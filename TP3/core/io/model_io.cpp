#include "io/model_io.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "io/json_value.hpp"

namespace tp3 {

void unflatten_into(const std::vector<double>& flat_weights,
                    const std::vector<std::size_t>& layer_sizes,
                    std::vector<Matrix>& weights_out,
                    std::vector<Matrix>& biases_out) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("unflatten_into: layer_sizes must contain at least 2 layers");
    }

    std::size_t expected_total = 0;
    for (std::size_t l = 0; l < layer_sizes.size() - 1; ++l) {
        expected_total += layer_sizes[l] * layer_sizes[l + 1] + layer_sizes[l + 1];
    }

    if (flat_weights.size() != expected_total) {
        throw std::invalid_argument("unflatten_into: flat_weights size mismatch: expected " +
                                    std::to_string(expected_total) + ", got " +
                                    std::to_string(flat_weights.size()));
    }

    weights_out.clear();
    biases_out.clear();
    weights_out.reserve(layer_sizes.size() - 1);
    biases_out.reserve(layer_sizes.size() - 1);

    std::size_t offset = 0;
    for (std::size_t l = 0; l < layer_sizes.size() - 1; ++l) {
        Matrix W(layer_sizes[l], layer_sizes[l + 1]);
        for (std::size_t r = 0; r < layer_sizes[l]; ++r) {
            for (std::size_t c = 0; c < layer_sizes[l + 1]; ++c) {
                W(r, c) = flat_weights[offset++];
            }
        }
        weights_out.push_back(std::move(W));

        Matrix b(1, layer_sizes[l + 1]);
        for (std::size_t c = 0; c < layer_sizes[l + 1]; ++c) {
            b(0, c) = flat_weights[offset++];
        }
        biases_out.push_back(std::move(b));
    }
}

std::filesystem::path save_model(const ModelSnapshot& snapshot, const std::filesystem::path& path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::vector<std::pair<std::string, JsonField>> fields;
    fields.emplace_back("format_version", JsonField::make_number(snapshot.run_config.format_version));
    fields.emplace_back("model_type", JsonField::make_string(snapshot.model_type));

    std::vector<double> sizes;
    for (auto s : snapshot.layer_sizes) sizes.push_back(static_cast<double>(s));
    fields.emplace_back("layer_sizes", JsonField::make_number_array(sizes));

    fields.emplace_back("activation", JsonField::make_string(snapshot.activation));
    fields.emplace_back("use_softmax_output", JsonField::make_bool(snapshot.use_softmax_output));
    fields.emplace_back("flat_weights", JsonField::make_number_array(snapshot.flat_weights));
    fields.emplace_back("epochs_completed", JsonField::make_number(snapshot.epochs_completed));
    fields.emplace_back("loss_per_epoch_so_far", JsonField::make_number_array(snapshot.loss_per_epoch_so_far));

    // RunConfig fields
    fields.emplace_back("loss", JsonField::make_string(snapshot.run_config.loss));
    fields.emplace_back("optimizer", JsonField::make_string(snapshot.run_config.optimizer_name));
    fields.emplace_back("momentum_coefficient", JsonField::make_number(snapshot.run_config.momentum_coefficient));
    fields.emplace_back("adam_beta1", JsonField::make_number(snapshot.run_config.adam_beta1));
    fields.emplace_back("adam_beta2", JsonField::make_number(snapshot.run_config.adam_beta2));
    fields.emplace_back("adam_epsilon", JsonField::make_number(snapshot.run_config.adam_epsilon));
    fields.emplace_back("learning_rate", JsonField::make_number(snapshot.run_config.learning_rate));
    fields.emplace_back("epochs", JsonField::make_number(snapshot.run_config.epochs));
    fields.emplace_back("seed", JsonField::make_number(static_cast<double>(snapshot.run_config.seed)));
    fields.emplace_back("dataset_kind", JsonField::make_string(snapshot.run_config.dataset_kind));
    if (!snapshot.run_config.dataset_case.empty()) {
        fields.emplace_back("dataset_case", JsonField::make_string(snapshot.run_config.dataset_case));
    }
    if (!snapshot.run_config.dataset_format.empty()) {
        fields.emplace_back("dataset_format", JsonField::make_string(snapshot.run_config.dataset_format));
    }
    if (!snapshot.run_config.dataset_path.empty()) {
        fields.emplace_back("dataset_path", JsonField::make_string(snapshot.run_config.dataset_path));
    }
    if (!snapshot.run_config.dataset_target_column.empty()) {
        fields.emplace_back("dataset_target_column", JsonField::make_string(snapshot.run_config.dataset_target_column));
    }
    fields.emplace_back("output_dir", JsonField::make_string(snapshot.run_config.output_dir));
    fields.emplace_back("progress_interval_epochs", JsonField::make_number(snapshot.run_config.progress_interval_epochs));
    if (!snapshot.run_config.resume_from.empty()) {
        fields.emplace_back("resume_from", JsonField::make_string(snapshot.run_config.resume_from));
    }

    std::string json_content = json_object_to_string(fields);

    std::filesystem::path tmp_path = path;
    tmp_path += ".tmp";

    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("save_model: cannot open " + tmp_path.string() + " for writing");
        }
        out << json_content;
        out.flush();
        if (!out) {
            throw std::runtime_error("save_model: failed writing to " + tmp_path.string());
        }
    }

    std::filesystem::rename(tmp_path, path);
    return path;
}

ModelSnapshot load_model(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("load_model: cannot open file " + path.string());
    }

    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string json_text = buffer.str();

    JsonObject obj = JsonObject::parse(json_text);

    std::string model_type = obj.get_string("model_type");
    if (model_type != "mlp" && model_type != "perceptron") {
        throw std::invalid_argument("load_model: unknown model_type: '" + model_type +
                                    "' (expected one of: mlp, perceptron)");
    }

    std::vector<double> sizes_d = obj.get_number_array("layer_sizes");
    if (sizes_d.size() < 2) {
        throw std::invalid_argument("load_model: layer_sizes must have at least 2 layers");
    }
    std::vector<std::size_t> layer_sizes;
    for (double s : sizes_d) {
        layer_sizes.push_back(static_cast<std::size_t>(s));
    }

    std::size_t expected_total = 0;
    for (std::size_t l = 0; l < layer_sizes.size() - 1; ++l) {
        expected_total += layer_sizes[l] * layer_sizes[l + 1] + layer_sizes[l + 1];
    }

    std::vector<double> flat_weights = obj.get_number_array("flat_weights");
    if (flat_weights.size() != expected_total) {
        throw std::invalid_argument("load_model: flat_weights size mismatch: expected " +
                                    std::to_string(expected_total) + ", got " +
                                    std::to_string(flat_weights.size()));
    }

    ModelSnapshot snapshot;
    snapshot.model_type = std::move(model_type);
    snapshot.layer_sizes = std::move(layer_sizes);
    snapshot.activation = obj.get_string("activation");
    snapshot.use_softmax_output = obj.get_bool_or("use_softmax_output", false);
    snapshot.flat_weights = std::move(flat_weights);
    snapshot.epochs_completed = static_cast<int>(obj.get_int_or("epochs_completed", 0));
    if (obj.has("loss_per_epoch_so_far")) {
        snapshot.loss_per_epoch_so_far = obj.get_number_array("loss_per_epoch_so_far");
    }
    snapshot.run_config = RunConfig::parse(json_text);

    return snapshot;
}

}  // namespace tp3
