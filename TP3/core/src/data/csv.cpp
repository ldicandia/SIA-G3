#include "data/csv.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tp3 {

namespace {

std::string trim_str(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

double parse_number_field(const std::string& field, const std::string& context) {
    std::string trimmed = trim_str(field);
    if (trimmed.empty()) {
        throw std::invalid_argument("empty numeric field in " + context);
    }
    char* endptr = nullptr;
    double val = std::strtod(trimmed.c_str(), &endptr);
    if (endptr != trimmed.c_str() + trimmed.size()) {
        throw std::invalid_argument("invalid numeric value '" + trimmed + "' in " + context);
    }
    return val;
}

}  // namespace

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    if (line.empty()) {
        return fields;
    }

    std::size_t i = 0;
    while (i < line.size()) {
        if (line[i] == '"') {
            // Quoted field
            ++i;
            std::string field;
            bool closed = false;
            while (i < line.size()) {
                if (line[i] == '"') {
                    if (i + 1 < line.size() && line[i + 1] == '"') {
                        field += '"';
                        i += 2;
                    } else {
                        closed = true;
                        ++i;
                        break;
                    }
                } else {
                    field += line[i++];
                }
            }
            if (!closed) {
                throw std::invalid_argument("unterminated quote in line: " + line);
            }
            if (i < line.size()) {
                if (line[i] != ',') {
                    throw std::invalid_argument("invalid character after closing quote in line: " + line);
                }
                ++i;
                if (i == line.size()) {
                    fields.push_back(field);
                    fields.push_back("");
                    return fields;
                }
            }
            fields.push_back(field);
        } else {
            // Unquoted field
            std::string field;
            while (i < line.size() && line[i] != ',') {
                field += line[i++];
            }
            fields.push_back(field);
            if (i < line.size() && line[i] == ',') {
                ++i;
                if (i == line.size()) {
                    fields.push_back("");
                }
            }
        }
    }
    return fields;
}

PlainCsv load_plain_csv(const std::filesystem::path& path, const std::string& target_column_name) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("load_plain_csv: cannot open file " + path.string());
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        return {Matrix(0, 0), Matrix(0, 0), {}};
    }
    // Handle CRLF
    if (!header_line.empty() && header_line.back() == '\r') {
        header_line.pop_back();
    }

    std::vector<std::string> header = parse_csv_line(header_line);
    if (header.empty()) {
        return {Matrix(0, 0), Matrix(0, 0), {}};
    }

    std::size_t target_idx = header.size();
    std::vector<std::string> feature_names;
    for (std::size_t j = 0; j < header.size(); ++j) {
        if (header[j] == target_column_name) {
            target_idx = j;
        } else {
            feature_names.push_back(header[j]);
        }
    }

    if (target_idx == header.size()) {
        throw std::invalid_argument("load_plain_csv: target column '" + target_column_name +
                                    "' not found in header of " + path.string());
    }

    const std::size_t num_features = header.size() - 1;
    std::vector<std::vector<double>> x_rows;
    std::vector<std::vector<double>> y_rows;

    std::string line;
    std::size_t row_num = 1;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> fields = parse_csv_line(line);
        if (fields.size() != header.size()) {
            throw std::invalid_argument("load_plain_csv: file " + path.string() + " row " +
                                        std::to_string(row_num) + ": expected " +
                                        std::to_string(header.size()) + " fields, got " +
                                        std::to_string(fields.size()));
        }

        std::vector<double> x_row;
        x_row.reserve(num_features);
        double y_val = 0.0;

        for (std::size_t j = 0; j < fields.size(); ++j) {
            std::string context = "file " + path.string() + " row " + std::to_string(row_num) +
                                  " column '" + header[j] + "'";
            double val = parse_number_field(fields[j], context);
            if (j == target_idx) {
                y_val = val;
            } else {
                x_row.push_back(val);
            }
        }

        x_rows.push_back(std::move(x_row));
        y_rows.push_back({y_val});
        ++row_num;
    }

    if (x_rows.empty()) {
        return {Matrix(0, num_features), Matrix(0, 1), feature_names};
    }

    return {Matrix::from_rows(x_rows), Matrix::from_rows(y_rows), feature_names};
}

LabeledImageCsv load_labeled_image_csv(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("load_labeled_image_csv: cannot open file " + path.string());
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        return {Matrix(0, 1), Matrix(0, 0)};
    }
    if (!header_line.empty() && header_line.back() == '\r') {
        header_line.pop_back();
    }

    std::vector<std::string> header = parse_csv_line(header_line);
    if (header.size() != 2 || header[0] != "label" || header[1] != "image") {
        throw std::invalid_argument("load_labeled_image_csv: expected header 'label,image', got '" +
                                    header_line + "'");
    }

    std::vector<double> labels_data;
    std::vector<double> images_data;
    std::size_t num_rows = 0;
    std::size_t image_length = 0;

    std::string line;
    std::size_t row_num = 1;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> fields = parse_csv_line(line);
        if (fields.size() != 2) {
            throw std::invalid_argument("load_labeled_image_csv: file " + path.string() + " row " +
                                        std::to_string(row_num) + ": expected 2 fields, got " +
                                        std::to_string(fields.size()));
        }

        std::string label_context = "file " + path.string() + " row " + std::to_string(row_num) + " label";
        double label_val = parse_number_field(fields[0], label_context);

        std::string img_str = trim_str(fields[1]);
        if (img_str.empty() || img_str.front() != '[' || img_str.back() != ']') {
            throw std::invalid_argument("load_labeled_image_csv: file " + path.string() + " row " +
                                        std::to_string(row_num) + " image must be enclosed in '[' and ']'");
        }

        std::string inner = img_str.substr(1, img_str.size() - 2);
        std::stringstream ss(inner);
        std::string item;
        std::size_t count = 0;

        while (std::getline(ss, item, ',')) {
            std::string item_context = "file " + path.string() + " row " + std::to_string(row_num) + " image item";
            double pix = parse_number_field(item, item_context);
            images_data.push_back(pix);
            ++count;
        }

        if (image_length == 0) {
            image_length = count;
        } else if (count != image_length) {
            throw std::invalid_argument("load_labeled_image_csv: file " + path.string() + " row " +
                                        std::to_string(row_num) + " image length mismatch: expected " +
                                        std::to_string(image_length) + ", got " + std::to_string(count));
        }

        labels_data.push_back(label_val);
        ++num_rows;
        ++row_num;
    }

    if (num_rows == 0) {
        return {Matrix(0, 1), Matrix(0, 0)};
    }

    Matrix labels_mat(num_rows, 1);
    for (std::size_t i = 0; i < num_rows; ++i) {
        labels_mat(i, 0) = labels_data[i];
    }

    Matrix images_mat(num_rows, image_length);
    for (std::size_t i = 0; i < num_rows; ++i) {
        for (std::size_t j = 0; j < image_length; ++j) {
            images_mat(i, j) = images_data[i * image_length + j];
        }
    }

    return {labels_mat, images_mat};
}

}  // namespace tp3
