"""End-to-end: the step perceptron learns AND (VAL-01), asserted from the run JSON (VAL-09)."""

from __future__ import annotations

EXPECTED_KEYS = {"case", "seed", "hyperparameters", "loss_per_epoch", "final_weights", "bias", "predictions"}
DATASET_ORDER = [[-1, 1], [1, -1], [-1, -1], [1, 1]]


def test_and_all_four_inputs_correct(run_validate) -> None:
    result = run_validate("and")
    predictions = result.data["predictions"]
    assert len(predictions) == 4
    assert all(p["predicted"] == p["expected"] for p in predictions)


def test_and_loss_history_has_one_entry_per_epoch_and_ends_at_zero(run_validate) -> None:
    result = run_validate("and")
    loss = result.data["loss_per_epoch"]
    assert len(loss) == result.data["hyperparameters"]["epochs"]
    assert loss[-1] == 0


def test_and_json_schema_and_dataset_order(run_validate) -> None:
    data = run_validate("and").data
    assert set(data) == EXPECTED_KEYS
    assert data["case"] == "and"
    assert data["seed"] == 42
    assert data["hyperparameters"]["activation"] == "step"
    assert len(data["final_weights"]) == 2
    assert [p["input"] for p in data["predictions"]] == DATASET_ORDER
