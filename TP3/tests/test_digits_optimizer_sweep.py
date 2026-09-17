import json
from pathlib import Path
import pytest
from scripts.ejercicio2.digits_optimizer_sweep import NEW_OPTIMIZER_VARIANTS


def test_optimizer_sweep_constants():
    assert "momentum" in NEW_OPTIMIZER_VARIANTS
    assert "adam" in NEW_OPTIMIZER_VARIANTS
    assert "sgd" not in NEW_OPTIMIZER_VARIANTS  # sgd is baseline


def test_optimizer_variants_json_structure_if_present():
    path = Path("docs/ejercicio2/optimizer_variants.json")
    if not path.exists():
        pytest.skip("optimizer_variants.json not yet generated")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vs = data.get("variants", [])
    assert len(vs) == 3
    opts = sorted(v["optimizer"] for v in vs)
    assert opts == ["adam", "momentum", "sgd"]
    assert all(v["val_accuracy"] > 0.10 for v in vs)


def test_optimizer_fairness_confound_guard():
    """Standing automated guard proving the three optimizer runs share architecture,

    learning_rate, and epochs. Fails loudly if any differs.
    """
    path = Path("docs/ejercicio2/optimizer_variants.json")
    if not path.exists():
        pytest.skip("optimizer_variants.json not yet generated")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vs = data.get("variants", [])
    assert len(vs) == 3

    archs = {tuple(v.get("architecture") or v.get("layer_sizes")) for v in vs}
    assert len(archs) == 1, f"Architecture differed across optimizer variants: {archs}"

    lrs = {v["learning_rate"] for v in vs}
    assert len(lrs) == 1, f"Learning rate differed across optimizer variants: {lrs}"

    epochs = {v["epochs"] for v in vs}
    assert len(epochs) == 1, f"Epochs differed across optimizer variants: {epochs}"
