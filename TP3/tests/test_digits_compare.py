import json
from pathlib import Path
import pytest
from scripts.digits_compare import select_best


def test_select_best_accuracy():
    variants = [
        {"name": "v1", "val_accuracy": 0.85},
        {"name": "v2", "val_accuracy": 0.92},
        {"name": "v3", "val_accuracy": 0.88},
    ]
    best = select_best(variants)
    assert best["name"] == "v2"


def test_select_best_tie_break_n_params():
    variants = [
        {
            "name": "large",
            "val_accuracy": 0.90,
            "n_params": 100,
            "learning_rate": 0.05,
            "optimizer": "sgd",
        },
        {
            "name": "compact",
            "val_accuracy": 0.90,
            "n_params": 50,
            "learning_rate": 0.05,
            "optimizer": "sgd",
        },
    ]
    best = select_best(variants)
    assert best["name"] == "compact"


def test_select_best_tie_break_learning_rate():
    variants = [
        {
            "name": "fast",
            "val_accuracy": 0.90,
            "n_params": 100,
            "learning_rate": 0.1,
            "optimizer": "sgd",
        },
        {
            "name": "slow",
            "val_accuracy": 0.90,
            "n_params": 100,
            "learning_rate": 0.01,
            "optimizer": "sgd",
        },
    ]
    best = select_best(variants)
    assert best["name"] == "slow"


def test_select_best_tie_break_optimizer():
    variants = [
        {
            "name": "v_sgd",
            "val_accuracy": 0.90,
            "n_params": 100,
            "learning_rate": 0.05,
            "optimizer": "sgd",
        },
        {
            "name": "v_adam",
            "val_accuracy": 0.90,
            "n_params": 100,
            "learning_rate": 0.05,
            "optimizer": "adam",
        },
    ]
    best = select_best(variants)
    assert best["name"] == "v_adam"


def test_variant_comparison_json_if_present():
    path = Path("docs/ejercicio2/variant_comparison.json")
    if not path.exists():
        pytest.skip("variant_comparison.json not yet generated")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "selected" in data
    sel = data["selected"]
    for k in ("layer_sizes", "learning_rate", "optimizer", "val_accuracy"):
        assert k in sel
    assert sel["val_accuracy"] > 0.10
