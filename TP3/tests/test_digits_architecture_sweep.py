import json
from pathlib import Path
import pytest
from scripts.digits_architecture_sweep import n_params, NEW_ARCH_VARIANTS


def test_n_params():
    # [2, 3, 1]: (2*3 + 3) + (3*1 + 1) = 9 + 4 = 13
    assert n_params([2, 3, 1]) == 13
    # [784, 16, 10]: (784*16 + 16) + (16*10 + 10) = 12560 + 170 = 12730
    assert n_params([784, 16, 10]) == 12730
    # [784, 32, 10]: (784*32 + 32) + (32*10 + 10) = 25120 + 330 = 25450
    assert n_params([784, 32, 10]) == 25450


def test_architecture_variants_json_if_present():
    path = Path("docs/ejercicio2/architecture_variants.json")
    if not path.exists():
        pytest.skip("architecture_variants.json not yet generated")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vs = data.get("variants", [])
    assert len(vs) == 3
    archs = sorted(tuple(v["layer_sizes"]) for v in vs)
    assert archs == [(784, 16, 10), (784, 32, 10), (784, 32, 16, 10)]

    # Controlled experiment: all fixed except architecture
    assert len({v["learning_rate"] for v in vs}) == 1
    assert len({v["optimizer"] for v in vs}) == 1
    assert len({v["epochs"] for v in vs}) == 1
    assert all(v["val_accuracy"] > 0.10 for v in vs)
