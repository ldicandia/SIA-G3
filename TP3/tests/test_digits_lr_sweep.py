import json
from pathlib import Path
import pytest
from scripts.ejercicio2.digits_lr_sweep import summarize_variant, NEW_LR_VARIANTS


def test_lr_sweep_configuration():
    assert 0.01 in NEW_LR_VARIANTS
    assert 2.0 in NEW_LR_VARIANTS
    assert 0.05 not in NEW_LR_VARIANTS  # 0.05 is baseline, reused


def test_lr_variants_json_if_present():
    path = Path("docs/ejercicio2/lr_variants.json")
    if not path.exists():
        pytest.skip("lr_variants.json not yet generated")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vs = data.get("variants", [])
    assert len(vs) == 3
    lrs = sorted(v["learning_rate"] for v in vs)
    assert lrs == [0.01, 0.05, 2.0]

    hi = next(v for v in vs if v["learning_rate"] == 2.0)
    base = next(v for v in vs if v["learning_rate"] == 0.05)
    lo = next(v for v in vs if v["learning_rate"] == 0.01)

    assert not lo["has_non_finite_loss"]
    assert not base["has_non_finite_loss"]
    # 2.0 is deliberately unstable or degraded
    assert hi["has_non_finite_loss"] or hi["val_accuracy"] <= base["val_accuracy"]
