#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "matrix.hpp"

namespace tp3 {

// Splits a CSV line into fields respecting double-quoted fields with embedded commas.
// Strips the enclosing quotes from quoted fields while preserving interior content.
std::vector<std::string> parse_csv_line(const std::string& line);

struct PlainCsv {
    Matrix X;
    Matrix y;
    std::vector<std::string> column_names;
};

// Loads a standard numeric CSV (e.g. fraud_dataset.csv) into feature matrix X and target y.
// column_names contains the header names of features in file order.
// Throws std::invalid_argument on missing target column, ragged rows, or non-numeric values.
PlainCsv load_plain_csv(const std::filesystem::path& path, const std::string& target_column_name);

struct LabeledImageCsv {
    Matrix labels;  // N x 1
    Matrix images;  // N x image_length (e.g. 784 for digits.csv)
};

// Loads labeled image datasets where image is a quoted bracketed array "[val1, val2, ...]".
// Validates header is exactly {"label", "image"} and all image rows have uniform length.
LabeledImageCsv load_labeled_image_csv(const std::filesystem::path& path);

}  // namespace tp3
