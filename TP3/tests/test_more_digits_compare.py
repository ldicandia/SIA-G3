import json
from pathlib import Path

import pytest

from scripts.ejercicio3.more_digits_compare import choose_final


def test_choose_final_ensemble_wins_strictly():
    best_individual = {
        "run_name": "arch_64-32",
        "layer_sizes": [784, 64, 32, 10],
        "learning_rate": 0.01,
        "optimizer": "sgd",
        "val_accuracy": 0.94,
    }
    ensemble = {
        "members": ["seed_7", "more_data_baseline", "seed_123", "seed_2026"],
        "ensemble_val_accuracy": 0.95,
    }
    result = choose_final(best_individual, ensemble)
    assert result["type"] == "ensemble"
    assert result["members"] == ensemble["members"]
    assert result["val_accuracy"] == 0.95


def test_choose_final_tie_prefers_single_model():
    best_individual = {
        "run_name": "arch_64-32",
        "layer_sizes": [784, 64, 32, 10],
        "learning_rate": 0.01,
        "optimizer": "sgd",
        "val_accuracy": 0.94,
    }
    ensemble = {
        "members": ["seed_7", "more_data_baseline", "seed_123", "seed_2026"],
        "ensemble_val_accuracy": 0.94,
    }
    result = choose_final(best_individual, ensemble)
    assert result["type"] == "single_model"
    assert result["run_name"] == "arch_64-32"
    assert result["layer_sizes"] == [784, 64, 32, 10]
    assert result["val_accuracy"] == 0.94


def test_choose_final_single_model_wins_when_ensemble_lower():
    best_individual = {
        "run_name": "arch_64-32",
        "layer_sizes": [784, 64, 32, 10],
        "learning_rate": 0.01,
        "optimizer": "sgd",
        "activation": "sigmoid",
        "epochs": 10,
        "val_accuracy": 0.9269,
    }
    ensemble = {
        "members": ["seed_7", "more_data_baseline", "seed_123", "seed_2026"],
        "ensemble_val_accuracy": 0.9266,
    }
    result = choose_final(best_individual, ensemble)
    assert result["type"] == "single_model"
    assert result["run_name"] == "arch_64-32"
    assert result["activation"] == "sigmoid"
    assert result["epochs"] == 10
    assert result["optimizer"] == "sgd"


def test_choose_final_single_model_defaults_activation_and_epochs():
    best_individual = {
        "run_name": "arch_64-32",
        "layer_sizes": [784, 64, 32, 10],
        "learning_rate": 0.01,
        "optimizer": "sgd",
        "val_accuracy": 0.9269,
    }
    ensemble = {
        "members": ["seed_7"],
        "ensemble_val_accuracy": 0.0,
    }
    result = choose_final(best_individual, ensemble)
    assert result["activation"] == "sigmoid"
    assert result["epochs"] == 10


def test_variant_comparison_json_if_present():
    path = Path("docs/ejercicio3/variant_comparison.json")
    if not path.exists():
        pytest.skip("variant_comparison.json not yet generated")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "architecture_variants" in data
    assert "seed_variants" in data
    assert "ensemble_variant" in data
    assert "best_individual" in data
    best = data["best_individual"]
    for k in ("layer_sizes", "learning_rate", "optimizer", "val_accuracy"):
        assert k in best

    final_choice = data["final_choice"]
    assert final_choice["type"] in ("single_model", "ensemble")
    assert final_choice["val_accuracy"] > 0.10
