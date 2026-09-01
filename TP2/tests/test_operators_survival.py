"""Contracts and tests for survival operators.

Pins the cátedra's additive survival strategy (selecting N from N + K union),
render accounting across generations (REP-08), pool ordering conventions,
and no-aliasing guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest

from tp2.engine.config import RunConfig
from tp2.engine.fitness import Evaluator
from tp2.engine.genome import Population
from tp2.engine.loop import Run
from tp2.engine.operators.crossover import make_one_point
from tp2.engine.operators.mutation import make_gene
from tp2.engine.operators.selection import make_elite
from tp2.engine.operators.survival import make_additive


def test_additive_survival_returns_top_n_of_union_disjoint_fitness():
    """Additive survival returns exactly the top N of the combined N + K pool."""
    n, k, l = 4, 4, 11
    # 4 parents with fitness [10, 30, 50, 70]
    parent_genes = np.ones((n, l), dtype=np.float32) * 1.0
    parent_fitness = np.array([10.0, 30.0, 50.0, 70.0], dtype=np.float32)
    parents = Population(parent_genes, parent_fitness)

    # 4 children with fitness [20, 40, 60, 80]
    child_genes = np.ones((k, l), dtype=np.float32) * 2.0
    child_fitness = np.array([20.0, 40.0, 60.0, 80.0], dtype=np.float32)
    children = Population(child_genes, child_fitness)

    survival = make_additive(replacement=make_elite())
    rng = np.random.default_rng(42)

    survivors = survival(parents, children, rng)

    assert survivors.genes.shape == (n, l)
    assert survivors.fitness.shape == (n,)
    # Top 4 fitness values across [10, 30, 50, 70, 20, 40, 60, 80] are 80, 70, 60, 50
    np.testing.assert_array_equal(survivors.fitness, [80.0, 70.0, 60.0, 50.0])


def test_additive_survival_render_accounting_unchanged_by_survival():
    """Survival call does not evaluate or increment evaluator renders (REP-08)."""
    target = np.zeros((16, 16, 3), dtype=np.uint8)
    evaluator = Evaluator(target=target, size=(16, 16))

    n, k, l = 4, 4, 11
    p_genes = np.zeros((n, l), dtype=np.float32)
    p_fit, _ = evaluator.evaluate_population(p_genes)
    assert evaluator.renders == n

    c_genes = np.ones((k, l), dtype=np.float32)
    c_fit, _ = evaluator.evaluate_population(c_genes)
    assert evaluator.renders == n + k

    renders_before_survival = evaluator.renders

    parents = Population(p_genes, p_fit)
    children = Population(c_genes, c_fit)
    survival = make_additive(replacement=make_elite())
    rng = np.random.default_rng(0)

    _ = survival(parents, children, rng)
    # Survival must not re-evaluate any individual
    assert evaluator.renders == renders_before_survival


def test_generation_loop_render_accounting_rises_by_child_count():
    """One full generation of the loop raises render count by exactly the child count."""
    target = np.zeros((16, 16, 3), dtype=np.uint8)
    evaluator = Evaluator(target=target, size=(16, 16))

    pop_size = 6
    child_count = 4
    config = RunConfig(
        population=pop_size,
        children=child_count,
        max_generations=5,
        recombination_probability=1.0,
        parents=make_elite(),
        replacement=make_elite(),
        crossover=make_one_point(),
        mutation=make_gene(0.1),
        survival=make_additive(replacement=make_elite()),
        effective={},
    )

    rng = np.random.default_rng(12345)
    run = Run(config=config, evaluator=evaluator, triangles=2, rng=rng)

    events = list(run)
    assert len(events) == 6  # gen 0 + 5 gens

    # Gen 0 has pop_size renders
    assert events[0].renders == pop_size

    # Every subsequent generation adds exactly child_count renders
    for g in range(1, 6):
        assert events[g].renders == events[g - 1].renders + child_count


def test_additive_survival_returns_copy_no_aliasing():
    """Survivor population's genes do not alias the pool array."""
    n, k, l = 2, 2, 11
    parents = Population(np.ones((n, l), dtype=np.float32) * 1.0, np.array([10.0, 20.0], dtype=np.float32))
    children = Population(np.ones((k, l), dtype=np.float32) * 2.0, np.array([30.0, 40.0], dtype=np.float32))

    survival = make_additive(replacement=make_elite())
    rng = np.random.default_rng(0)
    survivors = survival(parents, children, rng)

    # Modifying the returned genes does not modify parent or child arrays
    survivors.genes[0, 0] = 999.0
    assert parents.genes[0, 0] != 999.0
    assert children.genes[0, 0] != 999.0


def test_pool_ordering_convention_parents_first_tie_resolves_to_parent():
    """Parents occupy low indices (0..N-1) and children high ones (N..N+K-1).

    Under a stable sort with identical fitness, parents are selected before children.
    """
    n, k, l = 2, 2, 11
    # Parents and children have identical fitness
    parents = Population(np.full((n, l), 1.0, dtype=np.float32), np.array([50.0, 50.0], dtype=np.float32))
    children = Population(np.full((k, l), 2.0, dtype=np.float32), np.array([50.0, 50.0], dtype=np.float32))

    survival = make_additive(replacement=make_elite())
    rng = np.random.default_rng(0)
    survivors = survival(parents, children, rng)

    # Under stable sort with identical fitness, the first N individuals in union (the parents) are picked
    np.testing.assert_array_equal(survivors.genes, parents.genes)
