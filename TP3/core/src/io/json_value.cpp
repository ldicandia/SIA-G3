#include "io/json_value.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace tp3 {

namespace {

void skip_ws(const std::string& s, std::size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
        ++i;
    }
}

std::string parse_string(const std::string& s, std::size_t& i) {
    if (i >= s.size() || s[i] != '"') {
        throw std::invalid_argument("expected string at byte " + std::to_string(i));
    }
    ++i;
    std::string out;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\') {
            ++i;
            if (i >= s.size()) {
                throw std::invalid_argument("unterminated escape in string at byte " + std::to_string(i));
            }
            switch (s[i]) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                default:
                    out += s[i];
                    break;
            }
        } else {
            out += s[i];
        }
        ++i;
    }
    if (i >= s.size()) {
        throw std::invalid_argument("unterminated string literal");
    }
    ++i;  // skip closing quote
    return out;
}

double parse_number(const std::string& s, std::size_t& i) {
    const std::size_t start = i;
    if (i < s.size() && s[i] == '-') {
        ++i;
    }
    bool has_digits = false;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        has_digits = true;
        ++i;
    }
    if (!has_digits) {
        throw std::invalid_argument("invalid number format at byte " + std::to_string(start));
    }
    if (i < s.size() && s[i] == '.') {
        ++i;
        bool has_frac = false;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
            has_frac = true;
            ++i;
        }
        if (!has_frac) {
            throw std::invalid_argument("invalid fractional number at byte " + std::to_string(start));
        }
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
            ++i;
        }
        bool has_exp = false;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
            has_exp = true;
            ++i;
        }
        if (!has_exp) {
            throw std::invalid_argument("invalid exponent in number at byte " + std::to_string(start));
        }
    }

    std::string num_str = s.substr(start, i - start);
    char* endptr = nullptr;
    double val = std::strtod(num_str.c_str(), &endptr);
    if (endptr == num_str.c_str()) {
        throw std::invalid_argument("failed to parse number at byte " + std::to_string(start));
    }
    return val;
}

std::vector<double> parse_number_array(const std::string& s, std::size_t& i, const std::string& key) {
    if (i >= s.size() || s[i] != '[') {
        throw std::invalid_argument("expected '[' at byte " + std::to_string(i));
    }
    ++i;
    skip_ws(s, i);
    std::vector<double> arr;
    if (i < s.size() && s[i] == ']') {
        ++i;
        return arr;
    }

    while (i < s.size()) {
        skip_ws(s, i);
        if (i >= s.size()) break;
        if (s[i] == '-' || std::isdigit(static_cast<unsigned char>(s[i]))) {
            arr.push_back(parse_number(s, i));
        } else {
            throw std::invalid_argument("array for key '" + key + "' contains non-number at byte " + std::to_string(i));
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            skip_ws(s, i);
        } else if (i < s.size() && s[i] == ']') {
            ++i;
            return arr;
        } else {
            throw std::invalid_argument("expected ',' or ']' in array at byte " + std::to_string(i));
        }
    }
    throw std::invalid_argument("unterminated array for key '" + key + "'");
}

std::string number_to_str(double v) {
    if (!std::isfinite(v)) {
        return "null";
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

std::string quote_str(const std::string& s) {
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

}  // namespace

JsonObject JsonObject::parse(const std::string& text) {
    std::size_t i = 0;
    skip_ws(text, i);
    if (i >= text.size() || text[i] != '{') {
        throw std::invalid_argument("expected '{' at byte " + std::to_string(i));
    }
    ++i;

    JsonObject obj;
    skip_ws(text, i);
    if (i < text.size() && text[i] == '}') {
        ++i;
        skip_ws(text, i);
        if (i != text.size()) {
            throw std::invalid_argument("trailing characters after json object at byte " + std::to_string(i));
        }
        return obj;
    }

    while (i < text.size()) {
        skip_ws(text, i);
        if (i >= text.size() || text[i] != '"') {
            throw std::invalid_argument("expected string key at byte " + std::to_string(i));
        }
        std::string key = parse_string(text, i);
        skip_ws(text, i);
        if (i >= text.size() || text[i] != ':') {
            throw std::invalid_argument("expected ':' after key '" + key + "' at byte " + std::to_string(i));
        }
        ++i;
        skip_ws(text, i);
        if (i >= text.size()) {
            throw std::invalid_argument("unexpected end of json after key '" + key + "'");
        }

        JsonField field;
        if (text[i] == '{') {
            throw std::invalid_argument("nested object not allowed for key '" + key + "' at byte " + std::to_string(i));
        } else if (text[i] == '"') {
            field = JsonField::make_string(parse_string(text, i));
        } else if (text[i] == '[') {
            field = JsonField::make_number_array(parse_number_array(text, i, key));
        } else if (text[i] == 't' && text.compare(i, 4, "true") == 0) {
            i += 4;
            field = JsonField::make_bool(true);
        } else if (text[i] == 'f' && text.compare(i, 5, "false") == 0) {
            i += 5;
            field = JsonField::make_bool(false);
        } else if (text[i] == 'n' && text.compare(i, 4, "null") == 0) {
            i += 4;
            field = JsonField::make_null();
        } else if (text[i] == '-' || std::isdigit(static_cast<unsigned char>(text[i]))) {
            field = JsonField::make_number(parse_number(text, i));
        } else {
            throw std::invalid_argument("unexpected character at byte " + std::to_string(i) + " for key '" + key + "'");
        }

        obj.fields_[key] = std::move(field);

        skip_ws(text, i);
        if (i < text.size() && text[i] == ',') {
            ++i;
            skip_ws(text, i);
        } else if (i < text.size() && text[i] == '}') {
            ++i;
            break;
        } else {
            throw std::invalid_argument("expected ',' or '}' at byte " + std::to_string(i));
        }
    }

    skip_ws(text, i);
    if (i != text.size()) {
        throw std::invalid_argument("trailing characters after json object at byte " + std::to_string(i));
    }
    return obj;
}

bool JsonObject::has(const std::string& key) const {
    return fields_.find(key) != fields_.end();
}

std::string JsonObject::get_string(const std::string& key) const {
    auto it = fields_.find(key);
    if (it == fields_.end()) {
        throw std::invalid_argument("missing required string key '" + key + "'");
    }
    if (it->second.type != JsonFieldType::String) {
        throw std::invalid_argument("key '" + key + "' is not a string");
    }
    return it->second.string_value;
}

double JsonObject::get_number(const std::string& key) const {
    auto it = fields_.find(key);
    if (it == fields_.end()) {
        throw std::invalid_argument("missing required number key '" + key + "'");
    }
    if (it->second.type != JsonFieldType::Number) {
        throw std::invalid_argument("key '" + key + "' is not a number");
    }
    return it->second.number_value;
}

long long JsonObject::get_int(const std::string& key) const {
    double v = get_number(key);
    return static_cast<long long>(v);
}

bool JsonObject::get_bool(const std::string& key) const {
    auto it = fields_.find(key);
    if (it == fields_.end()) {
        throw std::invalid_argument("missing required bool key '" + key + "'");
    }
    if (it->second.type != JsonFieldType::Bool) {
        throw std::invalid_argument("key '" + key + "' is not a bool");
    }
    return it->second.bool_value;
}

std::vector<double> JsonObject::get_number_array(const std::string& key) const {
    auto it = fields_.find(key);
    if (it == fields_.end()) {
        throw std::invalid_argument("missing required number array key '" + key + "'");
    }
    if (it->second.type != JsonFieldType::NumberArray) {
        throw std::invalid_argument("key '" + key + "' is not a number array");
    }
    return it->second.array_value;
}

std::string JsonObject::get_string_or(const std::string& key, const std::string& fallback) const {
    if (!has(key)) return fallback;
    return get_string(key);
}

double JsonObject::get_number_or(const std::string& key, double fallback) const {
    if (!has(key)) return fallback;
    return get_number(key);
}

long long JsonObject::get_int_or(const std::string& key, long long fallback) const {
    if (!has(key)) return fallback;
    return get_int(key);
}

bool JsonObject::get_bool_or(const std::string& key, bool fallback) const {
    if (!has(key)) return fallback;
    return get_bool(key);
}

std::string json_object_to_string(const std::vector<std::pair<std::string, JsonField>>& fields) {
    std::string out = "{\n";
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const auto& pair = fields[i];
        out += "  " + quote_str(pair.first) + ": ";
        const JsonField& f = pair.second;
        switch (f.type) {
            case JsonFieldType::Null:
                out += "null";
                break;
            case JsonFieldType::Bool:
                out += (f.bool_value ? "true" : "false");
                break;
            case JsonFieldType::Number:
                out += number_to_str(f.number_value);
                break;
            case JsonFieldType::String:
                out += quote_str(f.string_value);
                break;
            case JsonFieldType::NumberArray: {
                out += "[";
                for (std::size_t j = 0; j < f.array_value.size(); ++j) {
                    if (j > 0) out += ", ";
                    out += number_to_str(f.array_value[j]);
                }
                out += "]";
                break;
            }
        }
        if (i + 1 < fields.size()) {
            out += ",\n";
        } else {
            out += "\n";
        }
    }
    out += "}\n";
    return out;
}

}  // namespace tp3
