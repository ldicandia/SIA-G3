"""Full-registry invariant tests for Phase 3 operator matrix.

Proves:
1. Registries contain at least 9 selection, 4 crossover, 4 mutation, and 3 survival operators.
2. Every registered operator round-trips through config validation and completes
   a real 2-generation run when placed in every slot it supports.
"""

from __future__ import annotations

import numpy as np
import pytest

from tp2.engine.config import build_run_config
from tp2.engine.fitness import Evaluator
from tp2.engine.loop import Run
from tp2.engine.operators.registry import CROSSOVER, MUTATION, SELECTION, SURVIVAL


def test_registry_minimum_counts():
    """Assert each registry has at least the full Phase 3 width."""
    assert len(SELECTION.names()) >= 9, f"Expected >=9 selection methods, got {SELECTION.names()}"
    assert len(CROSSOVER.names()) >= 4, f"Expected >=4 crossover methods, got {CROSSOVER.names()}"
    assert len(MUTATION.names()) >= 4, f"Expected >=4 mutation methods, got {MUTATION.names()}"
    assert len(SURVIVAL.names()) >= 3, f"Expected >=3 survival methods, got {SURVIVAL.names()}"


def _make_spec(kind: str, name: str) -> dict:
    """Provide minimal valid config spec for operator name."""
    if kind == "selection":
        if name == "boltzmann":
            return {"method": name, "t0": 10.0, "tc": 0.1, "k": 0.01}
        if name == "tournament_deterministic":
            return {"method": name, "m": 2}
        if name == "tournament_probabilistic":
            return {"method": name, "threshold": 0.75}
        if name == "blend":
            return {"method": name, "coefficient": 0.5, "method_1": {"method": "elite"}, "method_2": {"method": "random"}}
        return {"method": name}

    if kind == "crossover":
        if name == "uniform":
            return {"method": name, "p": 0.5, "boundary": "triangle"}
        return {"method": name, "boundary": "triangle"}

    if kind == "mutation":
        if name == "multigen_limited":
            return {"method": name, "m": 3, "probability": 0.5}
        return {"method": name, "probability": 0.5}

    if kind == "survival":
        if name == "generational_gap":
            return {"method": name, "gap": 0.5}
        return {"method": name}

    return {"method": name}


def _run_tiny(cfg_dict: dict) -> None:
    config = build_run_config(cfg_dict)
    evaluator = Evaluator(np.zeros((16, 16, 3), dtype=np.uint8), (16, 16))
    run = Run(config, evaluator, triangles=2, rng=np.random.default_rng(42))
    events = list(run)
    assert len(events) == 3  # gen 0, 1, 2
    assert run.result is not None
    assert run.result.stop_reason == "max_generations"


def _base_cfg() -> dict:
    return {
        "population": 4,
        "children": 4,
        "horizon": 2,
        "recombination_probability": 0.8,
        "parents": {"method": "elite"},
        "replacement": {"method": "elite"},
        "crossover": {"method": "one_point", "boundary": "triangle"},
        "mutation": {"method": "gene", "probability": 0.5},
        "survival": {"method": "additive"},
        "stop": {"max_generations": True},
    }


@pytest.mark.parametrize("sel_name", sorted(SELECTION.names()))
def test_all_selection_methods_run_in_parents_slot(sel_name: str):
    """Every selection operator can act as parent selection in a real run."""
    cfg = _base_cfg()
    cfg["parents"] = _make_spec("selection", sel_name)
    _run_tiny(cfg)


@pytest.mark.parametrize("sel_name", sorted(SELECTION.names()))
def test_all_selection_methods_run_in_replacement_slot(sel_name: str):
    """Every selection operator can act as replacement selection in a real run."""
    cfg = _base_cfg()
    cfg["replacement"] = _make_spec("selection", sel_name)
    _run_tiny(cfg)


@pytest.mark.parametrize("cross_name", sorted(CROSSOVER.names()))
def test_all_crossover_methods_run_in_crossover_slot(cross_name: str):
    """Every crossover operator runs in a real run."""
    cfg = _base_cfg()
    cfg["crossover"] = _make_spec("crossover", cross_name)
    _run_tiny(cfg)


@pytest.mark.parametrize("mut_name", sorted(MUTATION.names()))
def test_all_mutation_methods_run_in_mutation_slot(mut_name: str):
    """Every mutation operator runs in a real run."""
    cfg = _base_cfg()
    cfg["mutation"] = _make_spec("mutation", mut_name)
    _run_tiny(cfg)


@pytest.mark.parametrize("surv_name", sorted(SURVIVAL.names()))
def test_all_survival_methods_run_in_survival_slot(surv_name: str):
    """Every survival operator runs in a real run."""
    cfg = _base_cfg()
    cfg["survival"] = _make_spec("survival", surv_name)
    _run_tiny(cfg)
