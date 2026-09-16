"""End-to-end: MLP [2,2,1] and [2,3,2,1] learn XOR while step perceptron fails (VAL-04, VAL-05, VAL-06, VAL-07, VAL-09)."""

from __future__ import annotations

import json
from pathlib import Path

EXPECTED_KEYS = {"case", "seed", "hyperparameters", "loss_per_epoch", "final_weights", "bias", "predictions"}
DATASET_ORDER = [[-1, 1], [1, -1], [-1, -1], [1, 1]]


def test_xor_all_three_json_files_written(run_validate) -> None:
    result = run_validate("xor")
    out_dir = result.path.parent
    assert (out_dir / "xor_221.json").is_file()
    assert (out_dir / "xor_2321.json").is_file()
    assert (out_dir / "xor_step.json").is_file()


def test_xor_mlp_221_classifies_all_four_inputs_correctly(run_validate) -> None:
    result = run_validate("xor")
    path = result.path.parent / "xor_221.json"
    data = json.loads(path.read_text())

    assert data["case"] == "xor_221"
    assert data["hyperparameters"]["architecture"] == [2, 2, 1]
    assert data["hyperparameters"]["activation"] == "tanh"

    predictions = data["predictions"]
    assert len(predictions) == 4
    for p in predictions:
        class_pred = 1.0 if p["predicted"] >= 0.0 else -1.0
        assert class_pred == p["expected"]

    loss = data["loss_per_epoch"]
    assert len(loss) == data["hyperparameters"]["epochs"]
    assert loss[-1] < 0.05


def test_xor_mlp_2321_classifies_all_four_inputs_correctly(run_validate) -> None:
    result = run_validate("xor")
    path = result.path.parent / "xor_2321.json"
    data = json.loads(path.read_text())

    assert data["case"] == "xor_2321"
    assert data["hyperparameters"]["architecture"] == [2, 3, 2, 1]
    assert data["hyperparameters"]["activation"] == "tanh"

    predictions = data["predictions"]
    assert len(predictions) == 4
    for p in predictions:
        class_pred = 1.0 if p["predicted"] >= 0.0 else -1.0
        assert class_pred == p["expected"]

    loss = data["loss_per_epoch"]
    assert len(loss) == data["hyperparameters"]["epochs"]
    assert loss[-1] < 0.05


def test_xor_step_perceptron_fails_xor(run_validate) -> None:
    result = run_validate("xor")
    path = result.path.parent / "xor_step.json"
    data = json.loads(path.read_text())

    assert data["case"] == "xor_step"
    assert data["hyperparameters"]["activation"] == "step"

    predictions = data["predictions"]
    assert len(predictions) == 4

    correct = sum(1 for p in predictions if p["predicted"] == p["expected"])
    assert correct < 4  # Proves step perceptron fails XOR (VAL-06)

    loss = data["loss_per_epoch"]
    assert loss[-1] > 0.0


def test_xor_json_schema_and_dataset_order(run_validate) -> None:
    result = run_validate("xor")
    out_dir = result.path.parent

    for name in ["xor_221.json", "xor_2321.json", "xor_step.json"]:
        data = json.loads((out_dir / name).read_text())
        assert set(data) == EXPECTED_KEYS
        assert data["seed"] == 42
        assert [p["input"] for p in data["predictions"]] == DATASET_ORDER
