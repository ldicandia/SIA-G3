"""SC3 proof: multi-condition combined stop execution on a real run.

Proves:
1. A run with multiple stop conditions live terminates when the first condition triggers.
2. Only the final generation row carries the non-empty stop_reason naming the fired condition.
3. The outcome is deterministic and independent of dictionary / JSON source key order.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pytest

from tp2.engine.config import build_run_config
from tp2.engine.fitness import Evaluator
from tp2.engine.loop import Run
from tp2.io.images import load_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
ASSETS_DIR = PROJECT_ROOT / "assets"


def test_combined_stop_terminates_and_reports_single_fired_condition():
    """A run with 4 live conditions terminates and names exactly the winning condition."""
    cfg_json = json.loads((CONFIGS_DIR / "combined_stop_demo.json").read_text(encoding="utf-8"))
    target = load_target(ASSETS_DIR / "flag_ar.png", (32, 32))
    evaluator = Evaluator(target=target, size=(32, 32))

    config = build_run_config(cfg_json)
    run = Run(config, evaluator, triangles=10, rng=np.random.default_rng(42))

    started = time.perf_counter()
    events = list(run)
    elapsed = time.perf_counter() - started

    # Defensive bound against run hangs
    assert elapsed < 15.0, f"Combined stop run took too long: {elapsed:.2f}s"
    assert len(events) > 1, "Run should have executed at least 1 generation beyond gen 0"

    # Every row except last has empty stop reason
    for i, event in enumerate(events[:-1]):
        assert event.stop_reason == "", f"Generation {i} prematurely had stop_reason={event.stop_reason!r}"

    # Last row has exactly one of the four enabled conditions
    last_reason = events[-1].stop_reason
    enabled_reasons = {"wall_clock", "min_fitness", "content_stagnation", "structure_stagnation"}
    assert last_reason in enabled_reasons, f"Final stop_reason {last_reason!r} not in enabled set {enabled_reasons}"

    assert run.result is not None
    assert run.result.stop_reason == last_reason


def test_combined_stop_priority_is_independent_of_json_key_order():
    """Two identical configs differing only in dictionary key insertion order produce identical termination."""
    target = load_target(ASSETS_DIR / "flag_ar.png", (32, 32))
    evaluator = Evaluator(target=target, size=(32, 32))

    # Config A: min_fitness before content_stagnation
    cfg_a = {
        "population": 10,
        "children": 10,
        "horizon": 200,
        "recombination_probability": 0.8,
        "parents": {"method": "elite"},
        "replacement": {"method": "elite"},
        "crossover": {"method": "one_point"},
        "mutation": {"method": "gene", "probability": 0.5},
        "survival": {"method": "additive"},
        "stop": {
            "max_generations": False,
            "min_fitness": 0.8,
            "content_stagnation": {"window": 10, "tolerance": 0.001},
        },
    }

    # Config B: content_stagnation before min_fitness
    cfg_b = {
        "population": 10,
        "children": 10,
        "horizon": 200,
        "recombination_probability": 0.8,
        "parents": {"method": "elite"},
        "replacement": {"method": "elite"},
        "crossover": {"method": "one_point"},
        "mutation": {"method": "gene", "probability": 0.5},
        "survival": {"method": "additive"},
        "stop": {
            "content_stagnation": {"window": 10, "tolerance": 0.001},
            "max_generations": False,
            "min_fitness": 0.8,
        },
    }

    run_a = Run(build_run_config(cfg_a), evaluator, triangles=6, rng=np.random.default_rng(99))
    events_a = list(run_a)

    run_b = Run(build_run_config(cfg_b), evaluator, triangles=6, rng=np.random.default_rng(99))
    events_b = list(run_b)

    assert len(events_a) == len(events_b)
    assert events_a[-1].stop_reason == events_b[-1].stop_reason
    assert run_a.result.stop_reason == run_b.result.stop_reason
