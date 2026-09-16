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
