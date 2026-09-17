// ENG-07: to_json's format pinned byte for byte (fixed key order, 2-space
// indent, %.17g numbers, one prediction per line, trailing newline) -- this is
// what makes same-seed run files byte-identical.
#include <string>

#include "doctest.h"
#include "io/run_json.hpp"

TEST_CASE("to_json emits the exact expected string for a tiny record") {
    tp3::RunRecord r;
    r.case_name = "t";
    r.seed = 3;
    r.activation = "identity";
    r.learning_rate = 0.5;
    r.epochs = 2;
    r.n_inputs = 1;
    r.loss_per_epoch = {1.0, 0.25};
    r.final_weights = {0.5};
    r.bias = -1.0;
    r.predictions = {{{2.0}, 2.0, 1.5}};

    const std::string expected =
        "{\n"
        "  \"case\": \"t\",\n"
        "  \"seed\": 3,\n"
        "  \"hyperparameters\": {\n"
        "    \"activation\": \"identity\",\n"
        "    \"learning_rate\": 0.5,\n"
        "    \"epochs\": 2,\n"
        "    \"n_inputs\": 1\n"
        "  },\n"
        "  \"loss_per_epoch\": [1, 0.25],\n"
        "  \"final_weights\": [0.5],\n"
        "  \"bias\": -1,\n"
        "  \"predictions\": [\n"
        "    {\"input\": [2], \"expected\": 2, \"predicted\": 1.5}\n"
        "  ]\n"
        "}\n";

    CHECK(tp3::to_json(r) == expected);
}

TEST_CASE("to_json emits extended fields when populated") {
    tp3::RunRecord r;
    r.case_name = "t_ext";
    r.seed = 42;
    r.activation = "tanh";
    r.learning_rate = 0.1;
    r.epochs = 10;
    r.n_inputs = 2;
    r.loss_per_epoch = {0.5};
    r.final_weights = {0.1, 0.2};
    r.bias = 0.05;
    r.loss_name = "mse";
    r.optimizer_name = "sgd";
    r.dataset_path = "data/test.csv";
    r.wall_time_seconds = 1.234;
    r.predictions = {{{1.0, 2.0}, 1.0, 0.9}};

    const std::string json = tp3::to_json(r);
    CHECK(json.find("\"loss\": \"mse\"") != std::string::npos);
    CHECK(json.find("\"optimizer\": \"sgd\"") != std::string::npos);
    CHECK(json.find("\"dataset_path\": \"data/test.csv\"") != std::string::npos);
    CHECK(json.find("\"wall_time_seconds\": 1.234") != std::string::npos);
}

TEST_CASE("to_json emits predicted_class and expected_class when set") {
    tp3::RunRecord r;
    r.case_name = "t_mc";
    r.seed = 42;
    r.activation = "sigmoid";
    r.learning_rate = 0.05;
    r.epochs = 5;
    r.n_inputs = 784;
    r.loss_per_epoch = {0.8, 0.4};
    r.final_weights = {0.1};
    r.bias = 0.0;
    
    tp3::PredictionRecord pr;
    pr.input = {};
    pr.expected = 3.0;
    pr.predicted = 7.0;
    pr.predicted_class = 7;
    pr.expected_class = 3;
    r.predictions.push_back(pr);

    const std::string json = tp3::to_json(r);
    CHECK(json.find("{\"input\": [], \"expected\": 3, \"predicted\": 7, \"predicted_class\": 7, \"expected_class\": 3}") != std::string::npos);
}
