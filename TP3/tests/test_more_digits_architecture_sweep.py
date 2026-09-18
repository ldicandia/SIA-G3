import json
from pathlib import Path

import pytest

from scripts.ejercicio3.more_digits_architecture_sweep import (
    NEW_ARCH_VARIANTS,
    summarize_variant,
)


def _write_run_json(path, hyperparameters, optimizer, predictions):
    data = {
        "case": path.stem,
        "seed": 42,
        "hyperparameters": hyperparameters,
        "loss_per_epoch": [1.0, 0.5],
        "final_weights": [],
        "bias": 0.0,
        "optimizer": optimizer,
        "predictions": predictions,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_summarize_variant_reads_hyperparameters_and_computes_metrics(tmp_path):
    train_path = tmp_path / "arch_64_train.json"
    val_path = tmp_path / "arch_64_val.json"

    hp = {
        "activation": "sigmoid",
        "learning_rate": 0.01,
        "epochs": 10,
        "n_inputs": 784,
        "architecture": [784, 64, 10],
    }
    _write_run_json(train_path, hp, "sgd", [])

    predictions = [
        {"input": [], "expected": 0, "predicted": 0, "predicted_class": 0, "expected_class": 0},
        {"input": [], "expected": 1, "predicted": 1, "predicted_class": 1, "expected_class": 1},
        {"input": [], "expected": 2, "predicted": 3, "predicted_class": 3, "expected_class": 2},
    ]
    _write_run_json(val_path, hp, "sgd", predictions)

    result = summarize_variant("arch_64", [784, 64, 10], train_path, val_path)

    assert result["run_name"] == "arch_64"
    assert result["layer_sizes"] == [784, 64, 10]
    assert result["learning_rate"] == 0.01
    assert result["optimizer"] == "sgd"
    assert result["epochs"] == 10
    assert result["n_params"] == (784 * 64 + 64) + (64 * 10 + 10)
    assert result["val_accuracy"] == pytest.approx(2 / 3)
    assert result["train_run_json"] == str(train_path)
    assert result["val_run_json"] == str(val_path)


def test_new_arch_variants_are_distinct_from_baseline():
    assert [784, 32, 16, 10] not in NEW_ARCH_VARIANTS
    assert len(NEW_ARCH_VARIANTS) == 2
    assert [784, 64, 10] in NEW_ARCH_VARIANTS
    assert [784, 64, 32, 10] in NEW_ARCH_VARIANTS


def test_architecture_variants_json_if_present():
    """Standing guard: no confound, no silent drop of a worse-performing variant."""
    path = Path("docs/ejercicio3/architecture_variants.json")
    if not path.exists():
        pytest.skip("architecture_variants.json not yet generated")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vs = data.get("variants", [])
    assert len(vs) == 3, f"expected exactly 3 architecture variants, got {len(vs)}"

    archs = sorted(tuple(v["layer_sizes"]) for v in vs)
    assert archs == [(784, 32, 16, 10), (784, 64, 10), (784, 64, 32, 10)]

    lrs = {v["learning_rate"] for v in vs}
    assert len(lrs) == 1, f"learning_rate differed across entries: {lrs}"
    optimizers = {v["optimizer"] for v in vs}
    assert len(optimizers) == 1, f"optimizer differed across entries: {optimizers}"
    epochs = {v["epochs"] for v in vs}
    assert len(epochs) == 1, f"epochs differed across entries: {epochs}"

    assert all(v["val_accuracy"] > 0.10 for v in vs)
