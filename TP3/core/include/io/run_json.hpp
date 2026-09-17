#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace tp3 {

// One row of the expected-vs-predicted table.
struct PredictionRecord {
    std::vector<double> input;
    double expected;
    double predicted;
    int predicted_class = -1;
    int expected_class = -1;
};

// Everything one `tp3 validate` run stores for Python to analyse.
struct RunRecord {
    std::string case_name;
    unsigned long long seed;
    std::string activation;
    double learning_rate;
    int epochs;
    std::size_t n_inputs;
    std::vector<std::size_t> layer_sizes;
    std::vector<double> loss_per_epoch;
    std::vector<double> final_weights;
    double bias;
    std::vector<PredictionRecord> predictions;
    std::string dataset_path;
    double wall_time_seconds = 0.0;
    std::string loss_name;
    std::string optimizer_name;
};

// Pure: fixed key order, %.17g numbers (non-finite -> null), 2-space indent,
// one prediction object per line, trailing newline. Same record -> same bytes.
std::string to_json(const RunRecord& record);

// Creates out_dir, writes <out_dir>/<case_name>.json.tmp, renames it over
// <out_dir>/<case_name>.json and returns that path. Never leaves a partial file.
std::filesystem::path write_run_json(const RunRecord& record, const std::filesystem::path& out_dir);

}  // namespace tp3
