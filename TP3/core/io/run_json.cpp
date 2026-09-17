#include "io/run_json.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace tp3 {

namespace {

std::string number(double v) {
    if (!std::isfinite(v)) {
        return "null";
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

std::string quoted(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        out += c;
    }
    out += '"';
    return out;
}

std::string number_array(const std::vector<double>& values) {
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += number(values[i]);
    }
    out += "]";
    return out;
}

}  // namespace

std::string to_json(const RunRecord& r) {
    std::string out;
    out += "{\n";
    out += "  \"case\": " + quoted(r.case_name) + ",\n";
    out += "  \"seed\": " + std::to_string(r.seed) + ",\n";
    out += "  \"hyperparameters\": {\n";
    out += "    \"activation\": " + quoted(r.activation) + ",\n";
    out += "    \"learning_rate\": " + number(r.learning_rate) + ",\n";
    out += "    \"epochs\": " + std::to_string(r.epochs) + ",\n";
    out += "    \"n_inputs\": " + std::to_string(r.n_inputs);
    if (!r.layer_sizes.empty()) {
        out += ",\n    \"architecture\": [";
        for (std::size_t i = 0; i < r.layer_sizes.size(); ++i) {
            if (i > 0) {
                out += ", ";
            }
            out += std::to_string(r.layer_sizes[i]);
        }
        out += "]\n";
    } else {
        out += "\n";
    }
    out += "  },\n";
    out += "  \"loss_per_epoch\": " + number_array(r.loss_per_epoch) + ",\n";
    out += "  \"final_weights\": " + number_array(r.final_weights) + ",\n";
    out += "  \"bias\": " + number(r.bias) + ",\n";
    if (!r.loss_name.empty()) {
        out += "  \"loss\": " + quoted(r.loss_name) + ",\n";
    }
    if (!r.optimizer_name.empty()) {
        out += "  \"optimizer\": " + quoted(r.optimizer_name) + ",\n";
    }
    if (!r.dataset_path.empty()) {
        out += "  \"dataset_path\": " + quoted(r.dataset_path) + ",\n";
    }
    if (r.wall_time_seconds > 0.0) {
        out += "  \"wall_time_seconds\": " + number(r.wall_time_seconds) + ",\n";
    }
    out += "  \"predictions\": [\n";
    for (std::size_t i = 0; i < r.predictions.size(); ++i) {
        const PredictionRecord& p = r.predictions[i];
        out += "    {\"input\": " + number_array(p.input) + ", \"expected\": " + number(p.expected) +
               ", \"predicted\": " + number(p.predicted) + "}";
        out += (i + 1 < r.predictions.size()) ? ",\n" : "\n";
    }
    out += "  ]\n";
    out += "}\n";
    return out;
}

std::filesystem::path write_run_json(const RunRecord& record, const std::filesystem::path& out_dir) {
    std::filesystem::create_directories(out_dir);
    const std::filesystem::path final_path = out_dir / (record.case_name + ".json");
    const std::filesystem::path tmp_path = out_dir / (record.case_name + ".json.tmp");
    {
        std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
        if (!file) {
            throw std::runtime_error("write_run_json: cannot open " + tmp_path.string());
        }
        file << to_json(record);
        file.flush();
        if (!file) {
            throw std::runtime_error("write_run_json: write failed for " + tmp_path.string());
        }
    }
    std::filesystem::rename(tmp_path, final_path);
    return final_path;
}

}  // namespace tp3
