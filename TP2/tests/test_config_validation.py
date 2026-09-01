"""Phase-wide consolidated config validation and error rejection sweep (IO-02).

Proves that:
1. Malformed config files (empty object, non-object top levels) are rejected.
2. Every out-of-range value, invalid type, and unknown operator name is rejected
   BEFORE any evaluator rendering or allocation occurs.
3. Every rejection error message explicitly names the offending key or value.
"""

from __future__ import annotations

import copy
import json
import numpy as np
import pytest

from tp2.engine.config import ConfigError, build_run_config, load_config
from tp2.engine.fitness import Evaluator
from tp2.engine.loop import Run


def _valid_base() -> dict:
    return {
        "population": 4,
        "children": 4,
        "horizon": 10,
        "recombination_probability": 0.8,
        "parents": {"method": "elite"},
        "replacement": {"method": "elite"},
        "crossover": {"method": "one_point", "boundary": "triangle"},
        "mutation": {"method": "gene", "probability": 0.5},
        "survival": {"method": "additive"},
        "stop": {"max_generations": True},
    }


def test_load_config_rejects_non_object_top_level(tmp_path):
    """load_config rejects JSON array, scalar, or primitive at top level."""
    for bad_content in ("[1, 2, 3]", "\"string_root\"", "12345", "true"):
        p = tmp_path / "bad.json"
        p.write_text(bad_content, encoding="utf-8")
        with pytest.raises(ConfigError, match="must be an object"):
            load_config(p)


def test_build_run_config_rejects_empty_object():
    """An empty dict {} raises ConfigError naming missing keys."""
    with pytest.raises(ConfigError) as excinfo:
        build_run_config({})
    assert "population" in str(excinfo.value) or "horizon" in str(excinfo.value)


VALIDATION_CASES = [
    # Top-level numeric & type validations
    ("population is string", {"population": "10"}, "population"),
    ("population is 0", {"population": 0}, "population"),
    ("children is string", {"children": "4"}, "children"),
    ("children is 0", {"children": 0}, "children"),
    ("horizon is string", {"horizon": "100"}, "horizon"),
    ("horizon is 0", {"horizon": 0}, "horizon"),
    ("recombination_probability > 1.0", {"recombination_probability": 1.5}, "recombination_probability"),
    ("recombination_probability < 0.0", {"recombination_probability": -0.1}, "recombination_probability"),
    ("unknown top-level key", {"bogus_param": 42}, "bogus_param"),

    # Stop conditions validations
    ("stop is not object", {"stop": "max_generations"}, "stop"),
    ("stop empty object", {"stop": {}}, "stop"),
    ("stop all disabled", {"stop": {"max_generations": False}}, "stop"),
    ("stop wall_clock <= 0", {"stop": {"wall_clock_seconds": 0.0}}, "wall_clock_seconds"),
    ("stop min_fitness > 1.0", {"stop": {"min_fitness": 1.5}}, "min_fitness"),
    ("stop min_fitness <= 0.0", {"stop": {"min_fitness": 0.0}}, "min_fitness"),
    ("stop content_stagnation window 0", {"stop": {"content_stagnation": {"window": 0, "tolerance": 0.01}}}, "window"),
    ("stop structure_stagnation fraction > 1.0", {"stop": {"structure_stagnation": {"window": 3, "fraction": 1.5, "tolerance": 0.01}}}, "fraction"),

    # Operator parameters validations
    ("boltzmann tc <= 0", {"parents": {"method": "boltzmann", "t0": 10.0, "tc": 0.0}}, "tc"),
    ("tournament_probabilistic threshold < 0.5", {"parents": {"method": "tournament_probabilistic", "threshold": 0.3}}, "threshold"),
    ("tournament_deterministic m <= 0", {"parents": {"method": "tournament_deterministic", "m": 0}}, "m"),
    ("multigen_limited m <= 0", {"mutation": {"method": "multigen_limited", "m": 0, "probability": 0.5}}, "m"),
    ("mutation probability > 1.0", {"mutation": {"method": "gene", "probability": 1.5}}, "probability"),
    ("crossover boundary unknown", {"crossover": {"method": "one_point", "boundary": "invalid_boundary"}}, "boundary"),
    ("generational_gap gap < 0.0", {"survival": {"method": "generational_gap", "gap": -0.1}}, "gap"),
    ("generational_gap gap > 1.0", {"survival": {"method": "generational_gap", "gap": 1.2}}, "gap"),

    # Unknown operator names in each registry slot
    ("unknown selection method", {"parents": {"method": "nonexistent_sel"}}, "nonexistent_sel"),
    ("unknown replacement method", {"replacement": {"method": "nonexistent_rep"}}, "nonexistent_rep"),
    ("unknown crossover method", {"crossover": {"method": "nonexistent_cross"}}, "nonexistent_cross"),
    ("unknown mutation method", {"mutation": {"method": "nonexistent_mut"}}, "nonexistent_mut"),
    ("unknown survival method", {"survival": {"method": "nonexistent_surv"}}, "nonexistent_surv"),
]


@pytest.mark.parametrize("desc, override, expected_key", VALIDATION_CASES)
def test_config_validation_sweep_fails_before_render(desc: str, override: dict, expected_key: str):
    """Every malformed config fails before any render, explicitly naming the offending key."""
    cfg_data = _valid_base()
    cfg_data.update(override)

    evaluator = Evaluator(np.zeros((16, 16, 3), dtype=np.uint8), (16, 16))
    assert evaluator.renders == 0

    with pytest.raises(ConfigError) as excinfo:
        config = build_run_config(cfg_data)
        # If build_run_config unexpectedly succeeds, attempt to run 1 generation to check render accounting
        run = Run(config, evaluator, 2, np.random.default_rng(0))
        next(iter(run))
        pytest.fail(f"Validation for '{desc}' occurred too late: Evaluator executed {evaluator.renders} renders!")

    assert expected_key in str(excinfo.value), (
        f"For case '{desc}', expected error to name '{expected_key}', got: {excinfo.value}"
    )
    # Ensure zero renders happened
    assert evaluator.renders == 0
