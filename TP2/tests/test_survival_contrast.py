"""SC2 proof: exclusive survival non-monotonicity vs additive monotonicity.

Proves:
1. configs/exclusive_demo.json and configs/baseline.json differ in EXACTLY the survival slot.
2. An additive survival run maintains non-decreasing best_fitness across all generations.
3. An exclusive survival run with K > N exhibits at least one generation where best_fitness decreases.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tp2.engine.config import build_run_config
from tp2.engine.fitness import Evaluator
from tp2.engine.loop import Run
from tp2.io.images import load_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
ASSETS_DIR = PROJECT_ROOT / "assets"


def test_exclusive_and_baseline_configs_differ_only_in_survival_slot():
    """configs/exclusive_demo.json differs from configs/baseline.json in exactly survival."""
    base_path = CONFIGS_DIR / "baseline.json"
    excl_path = CONFIGS_DIR / "exclusive_demo.json"

    base_json = json.loads(base_path.read_text(encoding="utf-8"))
    excl_json = json.loads(excl_path.read_text(encoding="utf-8"))

    all_keys = set(base_json) | set(excl_json)
    diff_keys = [k for k in all_keys if base_json.get(k) != excl_json.get(k)]

    assert diff_keys == ["survival"], f"Expected difference in only 'survival', got {diff_keys}"
    assert base_json["survival"] == {"method": "additive"}
    assert excl_json["survival"] == {"method": "exclusive"}
    assert base_json["children"] > base_json["population"], "K > N must hold for exclusive contrast"


def test_additive_monotone_vs_exclusive_non_monotone_contrast():
    """Controlled one-variable run proves additive is monotone and exclusive has dips."""
    target_raw = load_target(ASSETS_DIR / "flag_ar.png", (32, 32))
    evaluator = Evaluator(target=target_raw, size=(32, 32))
    triangles = 10
    seed = 1
    gens = 120

    # 1. Additive run
    base_json = json.loads((CONFIGS_DIR / "baseline.json").read_text(encoding="utf-8"))
    base_json["horizon"] = gens
    base_json["population"] = 15
    base_json["children"] = 25  # K > N
    cfg_add = build_run_config(base_json)
    run_add = Run(cfg_add, evaluator, triangles, np.random.default_rng(seed))
    add_fitness = [e.best_fitness for e in run_add]

    # Additive must be strictly non-decreasing across all rows
    for i in range(1, len(add_fitness)):
        assert add_fitness[i] >= add_fitness[i - 1] - 1e-6, (
            f"Additive survival decreased at generation {i}: {add_fitness[i-1]} -> {add_fitness[i]}"
        )

    # 2. Exclusive run (identical seed, target, parameters except survival slot)
    excl_json = json.loads((CONFIGS_DIR / "exclusive_demo.json").read_text(encoding="utf-8"))
    excl_json["horizon"] = gens
    excl_json["population"] = 15
    excl_json["children"] = 25  # K > N
    cfg_excl = build_run_config(excl_json)
    run_excl = Run(cfg_excl, evaluator, triangles, np.random.default_rng(seed))
    excl_fitness = [e.best_fitness for e in run_excl]

    # Exclusive must show at least one dip (strictly less than previous generation)
    has_dip = any(excl_fitness[i] < excl_fitness[i - 1] - 1e-7 for i in range(1, len(excl_fitness)))
    assert has_dip, "Exclusive survival with K > N was expected to have at least one fitness dip"
