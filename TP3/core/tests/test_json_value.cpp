#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"
#include "io/json_value.hpp"

using tp3::JsonField;
using tp3::JsonObject;
using tp3::json_object_to_string;

TEST_CASE("JsonObject: empty, basic parsing, and accessors") {
    JsonObject empty = JsonObject::parse("{}");
    CHECK(empty.fields().empty());
    CHECK_FALSE(empty.has("anything"));

    CHECK_THROWS_AS(JsonObject::parse(""), std::invalid_argument);
    CHECK_THROWS_AS(JsonObject::parse("{not json"), std::invalid_argument);
    CHECK_THROWS_AS(JsonObject::parse("{\"key\": \"val\""), std::invalid_argument);
    CHECK_THROWS_AS(JsonObject::parse("{\"key\": \"val\"} trailing"), std::invalid_argument);

    std::string json = "{\n"
                       "  \"a\": 1,\n"
                       "  \"b\": \"x\",\n"
                       "  \"c\": true,\n"
                       "  \"d\": 2.5,\n"
                       "  \"arr\": [64, 32, 10]\n"
                       "}";
    JsonObject obj = JsonObject::parse(json);
    CHECK(obj.get_int("a") == 1);
    CHECK(obj.get_string("b") == "x");
    CHECK(obj.get_bool("c") == true);
    CHECK(obj.get_number("d") == doctest::Approx(2.5));

    std::vector<double> expected_arr = {64.0, 32.0, 10.0};
    CHECK(obj.get_number_array("arr") == expected_arr);

    CHECK(obj.get_number_or("missing", 3.5) == doctest::Approx(3.5));
    CHECK(obj.get_string_or("missing", "def") == "def");
    CHECK(obj.get_int_or("missing", 42) == 42);
    CHECK(obj.get_bool_or("missing", false) == false);
    CHECK_FALSE(obj.has("missing"));
}

TEST_CASE("JsonObject: rejects nested objects and non-number arrays") {
    CHECK_THROWS_AS(JsonObject::parse("{\"nested\": {\"a\": 1}}"), std::invalid_argument);
    CHECK_THROWS_AS(JsonObject::parse("{\"str_arr\": [\"a\", \"b\"]}"), std::invalid_argument);
}

TEST_CASE("json_object_to_string: formats properly") {
    std::vector<std::pair<std::string, JsonField>> fields = {
        {"name", JsonField::make_string("mlp")},
        {"lr", JsonField::make_number(0.1)},
        {"active", JsonField::make_bool(true)},
        {"layers", JsonField::make_number_array({2.0, 1.0})}
    };
    std::string s = json_object_to_string(fields);
    JsonObject parsed = JsonObject::parse(s);
    CHECK(parsed.get_string("name") == "mlp");
    CHECK(parsed.get_number("lr") == doctest::Approx(0.1));
    CHECK(parsed.get_bool("active") == true);
    CHECK(parsed.get_number_array("layers") == std::vector<double>{2.0, 1.0});
}
