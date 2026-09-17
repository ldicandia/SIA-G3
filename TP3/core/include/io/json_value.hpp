#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tp3 {

enum class JsonFieldType {
    Null,
    Bool,
    Number,
    String,
    NumberArray,
};

struct JsonField {
    JsonFieldType type = JsonFieldType::Null;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<double> array_value;

    static JsonField make_null() { return {}; }
    static JsonField make_bool(bool v) {
        JsonField f;
        f.type = JsonFieldType::Bool;
        f.bool_value = v;
        return f;
    }
    static JsonField make_number(double v) {
        JsonField f;
        f.type = JsonFieldType::Number;
        f.number_value = v;
        return f;
    }
    static JsonField make_string(std::string v) {
        JsonField f;
        f.type = JsonFieldType::String;
        f.string_value = std::move(v);
        return f;
    }
    static JsonField make_number_array(std::vector<double> v) {
        JsonField f;
        f.type = JsonFieldType::NumberArray;
        f.array_value = std::move(v);
        return f;
    }
};

// Restricted JSON parser and container for flat key-value configs and snapshots.
// Rejects nested objects and arrays of non-numbers.
class JsonObject {
public:
    static JsonObject parse(const std::string& text);

    bool has(const std::string& key) const;

    std::string get_string(const std::string& key) const;
    double get_number(const std::string& key) const;
    long long get_int(const std::string& key) const;
    bool get_bool(const std::string& key) const;
    std::vector<double> get_number_array(const std::string& key) const;

    std::string get_string_or(const std::string& key, const std::string& fallback) const;
    double get_number_or(const std::string& key, double fallback) const;
    long long get_int_or(const std::string& key, long long fallback) const;
    bool get_bool_or(const std::string& key, bool fallback) const;

    const std::unordered_map<std::string, JsonField>& fields() const { return fields_; }

private:
    std::unordered_map<std::string, JsonField> fields_;
};

// Formats restricted JSON object fields with 2-space indentation.
std::string json_object_to_string(const std::vector<std::pair<std::string, JsonField>>& fields);

}  // namespace tp3
