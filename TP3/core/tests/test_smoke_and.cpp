#include <random>
#include <string>

#include "doctest.h"
#include "io/run_json.hpp"
#include "perceptron.hpp"
#include "validation/datasets.hpp"

using tp3::Matrix;

TEST_CASE("step perceptron learns AND with seed 42") {
    std::mt19937_64 rng(42);
    const tp3::Dataset data = tp3::and_dataset();
    tp3::SimplePerceptron model(2, "step", 0.1, rng);

    const tp3::TrainResult result = model.fit(data.X, data.y, 100);

    CHECK(result.loss_per_epoch.size() == 100);
    CHECK(result.loss_per_epoch.back() == 0.0);
    CHECK(model.predict(data.X) == data.y);
}

TEST_CASE("to_json is deterministic and carries the schema keys") {
    tp3::RunRecord record;
    record.case_name = "and";
    record.seed = 42;
    record.activation = "step";
    record.learning_rate = 0.1;
    record.epochs = 3;
    record.n_inputs = 2;
    record.loss_per_epoch = {1.0, 0.5, 0.0};
    record.final_weights = {0.25, -0.125};
    record.bias = 0.0625;
    record.predictions = {{{-1.0, 1.0}, -1.0, -1.0}, {{1.0, 1.0}, 1.0, 1.0}};

    const std::string a = tp3::to_json(record);
    const std::string b = tp3::to_json(record);

    CHECK(a == b);
    CHECK(a.find("\"case\": \"and\"") != std::string::npos);
    CHECK(a.find("\"loss_per_epoch\": [") != std::string::npos);
    CHECK(a.find("\"final_weights\": [") != std::string::npos);
    CHECK(a.find("\"bias\":") != std::string::npos);
    CHECK(a.find("\"predictions\": [") != std::string::npos);
}
