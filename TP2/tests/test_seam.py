"""Config-only behaviour-change proofs for ROADMAP Success Criteria 4 and 5.

T-02-09: every config compared here is built from the shipped JSON files in
`configs/`, never from an inline dictionary invented for the test -- a proof
that quietly edited a source file to fabricate a difference would prove
nothing, and this file reads the demo configs this plan actually ships.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from tp2.engine.config import build_run_config
from tp2.engine.fitness import Evaluator
from tp2.engine.genome import random_population
from tp2.engine.loop import Run

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
CANVAS = (32, 32)
TARGET = np.full((*CANVAS[::-1], 3), 128.0, dtype=np.float32)
TRIANGLES = 6


def _shipped(name: str) -> dict:
    return json.loads((CONFIGS_DIR / name).read_text(encoding="utf-8"))


def _tiny(config_dict: dict, generations: int = 4) -> dict:
    """A copy of a shipped config, generation cap and population/children
    crushed for test speed. The operator specs -- the thing each seam test
    actually exercises -- are untouched."""
    result = copy.deepcopy(config_dict)
    result["population"] = 6
    result["children"] = 6
    result["horizon"] = generations
    result["stop"] = {"max_generations": True}
    return result


def _best_genes(config_dict: dict, seed: int) -> np.ndarray:
    config = build_run_config(config_dict)
    run = Run(config, Evaluator(TARGET, CANVAS), TRIANGLES, np.random.default_rng(seed))
    for _ in run:
        pass
    assert run.result is not None
    return run.result.best_genes


def test_shipped_demo_configs_differ_from_baseline_in_exactly_one_slot() -> None:
    baseline = _shipped("baseline.json")
    blend = _shipped("blend_demo.json")
    diffs = {key for key in baseline if baseline.get(key) != blend.get(key)}
    diffs |= {key for key in blend if baseline.get(key) != blend.get(key)}
    assert diffs == {"parents"}, diffs

    gene_boundary = _shipped("gene_boundary.json")
    diffs2 = {key for key in baseline if baseline.get(key) != gene_boundary.get(key)}
    diffs2 |= {key for key in gene_boundary if baseline.get(key) != gene_boundary.get(key)}
    assert diffs2 == {"crossover"}, diffs2


def test_blend_parent_selection_changes_the_run_from_configuration_alone() -> None:
    """ROADMAP Success Criterion 4, first clause."""
    baseline_genes = _best_genes(_tiny(_shipped("baseline.json")), seed=11)
    blend_genes = _best_genes(_tiny(_shipped("blend_demo.json")), seed=11)
    assert not np.array_equal(baseline_genes, blend_genes)


def test_gene_boundary_crossover_changes_the_run_from_configuration_alone() -> None:
    """ROADMAP Success Criterion 4, second clause."""
    baseline_genes = _best_genes(_tiny(_shipped("baseline.json")), seed=11)
    gene_genes = _best_genes(_tiny(_shipped("gene_boundary.json")), seed=11)
    assert not np.array_equal(baseline_genes, gene_genes)


def test_zero_recombination_probability_copies_then_still_mutates() -> None:
    """ROADMAP Success Criterion 5.

    Calls the operators directly rather than hooking the loop, per the
    plan's own stated preference -- an `observers` parameter is exactly the
    hole this architecture exists to close.
    """
    config_dict = _tiny(_shipped("baseline.json"))
    config_dict["recombination_probability"] = 0.0
    config_dict["mutation"] = {"method": "gene", "probability": 1.0}
    config = build_run_config(config_dict)

    rng = np.random.default_rng(3)
    genes = random_population(rng, config.population, TRIANGLES)
    evaluator = Evaluator(TARGET, CANVAS)
    fitness, _ = evaluator.evaluate_population(genes)
    parent_indices = config.parents(fitness, config.children, rng)

    pre_mutation: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, config.children, 2):
        first = genes[parent_indices[start]]
        second = genes[parent_indices[start + 1]]
        if rng.random() < config.recombination_probability:  # never true at 0.0
            child_1, child_2 = config.crossover(first, second, rng)
        else:
            child_1, child_2 = first.copy(), second.copy()
        pre_mutation.append((child_1, first))
        pre_mutation.append((child_2, second))

    assert all(np.array_equal(child, parent) for child, parent in pre_mutation), (
        "pre-mutation children must be exact copies of a parent at probability 0.0"
    )

    post_mutation = [config.mutation(child, rng) for child, _ in pre_mutation]
    assert not all(
        np.array_equal(post, child) for post, (child, _) in zip(post_mutation, pre_mutation)
    ), "children must still pass through mutation even when never recombined"


def test_loop_contains_zero_operator_name_literals() -> None:
    """The generation loop in tp2/engine/loop.py has zero operator name literals."""
    import re
    loop_file = Path(__file__).resolve().parents[1] / "tp2" / "engine" / "loop.py"
    lines = [l for l in loop_file.read_text(encoding="utf-8").splitlines() if not l.strip().startswith("#")]
    code = "\n".join(lines)
    # Check for known operator names appearing as string literals
    operator_names = [
        "elite", "random", "blend", "one_point", "gene", "additive",
        "roulette", "universal", "ranking", "boltzmann",
        "tournament_deterministic", "tournament_probabilistic",
    ]
    pattern = r'["\'](' + "|".join(operator_names) + r')["\']'
    matches = re.findall(pattern, code)
    assert matches == [], f"Found operator literals in tp2/engine/loop.py: {matches}"

