"""Contracts and tests for survival operators.

Pins:
- SUR-01: Additive survival (selecting N from N + K union)
- SUR-02, SUR-03: Exclusive survival with strict K > N branch boundary and K <= N infill
- SUR-04: Replacement selection parameterization
- SUR-05: Brecha Generacional G splitting N into (1-G)*N parents and G*N children
- REP-08: Render accounting across generations and zero re-evaluation in survival
"""

from __future__ import annotations

import numpy as np
import pytest

from tp2.engine.config import ConfigError, RunConfig, build_run_config
from tp2.engine.fitness import Evaluator
from tp2.engine.genome import Population
from tp2.engine.loop import Run
from tp2.engine.operators.crossover import make_one_point
from tp2.engine.operators.mutation import make_gene
from tp2.engine.operators.registry import SURVIVAL
from tp2.engine.operators.selection import make_elite
from tp2.engine.operators.survival import (
    make_additive,
    make_exclusive,
    make_generational_gap,
)


def test_survival_registry_contains_all_three_strategies():
    """SURVIVAL registry contains additive, exclusive, and generational_gap."""
    names = SURVIVAL.names()
    assert {"additive", "exclusive", "generational_gap"} <= set(names)


def test_additive_survival_returns_top_n_of_union_disjoint_fitness():
    """Additive survival returns exactly the top N of the combined N + K pool."""
    n, k, l = 4, 4, 11
    parent_genes = np.ones((n, l), dtype=np.float32) * 1.0
    parent_fitness = np.array([10.0, 30.0, 50.0, 70.0], dtype=np.float32)
    parents = Population(parent_genes, parent_fitness)

    child_genes = np.ones((k, l), dtype=np.float32) * 2.0
    child_fitness = np.array([20.0, 40.0, 60.0, 80.0], dtype=np.float32)
    children = Population(child_genes, child_fitness)

    survival = make_additive(replacement=make_elite())
    rng = np.random.default_rng(42)

    survivors = survival(parents, children, rng)

    assert survivors.genes.shape == (n, l)
    assert survivors.fitness.shape == (n,)
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
    assert evaluator.renders == renders_before_survival


def test_generation_loop_render_accounting_rises_by_child_count():
    """One full generation of the loop raises render count by exactly the child count."""
    target = np.zeros((16, 16, 3), dtype=np.uint8)
    evaluator = Evaluator(target=target, size=(16, 16))

    pop_size = 6
    child_count = 4
    config = build_run_config({
        "population": pop_size,
        "children": child_count,
        "horizon": 5,
        "recombination_probability": 1.0,
        "parents": {"method": "elite"},
        "replacement": {"method": "elite"},
        "crossover": {"method": "one_point"},
        "mutation": {"method": "gene", "probability": 0.1},
        "survival": {"method": "additive"},
        "stop": {"max_generations": True},
    })

    rng = np.random.default_rng(12345)
    run = Run(config=config, evaluator=evaluator, triangles=2, rng=rng)

    events = list(run)
    assert len(events) == 6  # gen 0 + 5 gens

    assert events[0].renders == pop_size

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

    survivors.genes[0, 0] = 999.0
    assert parents.genes[0, 0] != 999.0
    assert children.genes[0, 0] != 999.0


def test_pool_ordering_convention_parents_first_tie_resolves_to_parent():
    """Parents occupy low indices (0..N-1) and children high ones (N..N+K-1)."""
    n, k, l = 2, 2, 11
    parents = Population(np.full((n, l), 1.0, dtype=np.float32), np.array([50.0, 50.0], dtype=np.float32))
    children = Population(np.full((k, l), 2.0, dtype=np.float32), np.array([50.0, 50.0], dtype=np.float32))

    survival = make_additive(replacement=make_elite())
    rng = np.random.default_rng(0)
    survivors = survival(parents, children, rng)

    np.testing.assert_array_equal(survivors.genes, parents.genes)


# Task 1: Exclusive survival


def test_exclusive_survival_k_greater_than_n():
    """Exclusive with K > N selects N individuals exclusively from children."""
    n = 20
    k = 40
    l = 11

    # Parents (100..119) vs Children (1..40)
    # Even if parents have higher fitness, they MUST NOT survive
    parent_genes = np.ones((n, l), dtype=np.float32) * 1.0
    parent_fitness = np.arange(100, 100 + n, dtype=np.float32)
    parents = Population(parent_genes, parent_fitness)

    child_genes = np.ones((k, l), dtype=np.float32) * 2.0
    child_fitness = np.arange(1, 1 + k, dtype=np.float32)
    children = Population(child_genes, child_fitness)

    survival = make_exclusive(replacement=make_elite())
    rng = np.random.default_rng(42)
    survivors = survival(parents, children, rng)

    assert survivors.genes.shape == (n, l)
    # All survivors are children (genes == 2.0), no parents (genes == 1.0)
    assert np.all(survivors.genes == 2.0)
    # Elite selection from children: top 20 children have fitness 21..40
    expected_fit = np.arange(40, 20, -1, dtype=np.float32)
    np.testing.assert_array_equal(survivors.fitness, expected_fit)


def test_exclusive_survival_k_less_than_n():
    """Exclusive with K < N returns all K children plus (N - K) parents."""
    n = 20
    k = 10
    l = 11

    parent_genes = np.ones((n, l), dtype=np.float32) * 1.0
    parent_fitness = np.arange(1, 1 + n, dtype=np.float32)  # 1..20
    parents = Population(parent_genes, parent_fitness)

    child_genes = np.ones((k, l), dtype=np.float32) * 2.0
    child_fitness = np.arange(50, 50 + k, dtype=np.float32)  # 50..59
    children = Population(child_genes, child_fitness)

    survival = make_exclusive(replacement=make_elite())
    rng = np.random.default_rng(0)
    survivors = survival(parents, children, rng)

    assert survivors.genes.shape == (n, l)
    # First K=10 are all children (genes == 2.0)
    assert np.all(survivors.genes[:k] == 2.0)
    np.testing.assert_array_equal(survivors.fitness[:k], child_fitness)

    # Next N-K=10 are top parents (genes == 1.0, fitness 20..11)
    assert np.all(survivors.genes[k:] == 1.0)
    expected_p_fit = np.arange(20, 10, -1, dtype=np.float32)
    np.testing.assert_array_equal(survivors.fitness[k:], expected_p_fit)


def test_exclusive_survival_k_equals_n_boundary():
    """Exclusive with K == N returns all K children and 0 parents."""
    n = 20
    k = 20
    l = 11

    parent_genes = np.ones((n, l), dtype=np.float32) * 1.0
    parent_fitness = np.full(n, 100.0, dtype=np.float32)
    parents = Population(parent_genes, parent_fitness)

    child_genes = np.ones((k, l), dtype=np.float32) * 2.0
    child_fitness = np.full(k, 10.0, dtype=np.float32)
    children = Population(child_genes, child_fitness)

    survival = make_exclusive(replacement=make_elite())
    rng = np.random.default_rng(0)
    survivors = survival(parents, children, rng)

    assert survivors.genes.shape == (n, l)
    assert np.all(survivors.genes == 2.0)
    np.testing.assert_array_equal(survivors.fitness, child_fitness)


def test_exclusive_survival_k_zero():
    """Exclusive with K == 0 returns N parents via replacement selection."""
    n = 20
    k = 0
    l = 11

    parents = Population(np.ones((n, l), dtype=np.float32), np.arange(1, n + 1, dtype=np.float32))
    children = Population(np.zeros((0, l), dtype=np.float32), np.zeros(0, dtype=np.float32))

    survival = make_exclusive(replacement=make_elite())
    rng = np.random.default_rng(0)
    survivors = survival(parents, children, rng)

    assert survivors.genes.shape == (n, l)
    expected_fit = np.arange(n, 0, -1, dtype=np.float32)
    np.testing.assert_array_equal(survivors.fitness, expected_fit)


def test_exclusive_survival_render_accounting_unchanged():
    """Exclusive survival consumes zero renders under both K > N and K <= N branches."""
    target = np.zeros((16, 16, 3), dtype=np.uint8)
    evaluator = Evaluator(target=target, size=(16, 16))

    n, l = 10, 11
    p_genes = np.zeros((n, l), dtype=np.float32)
    p_fit, _ = evaluator.evaluate_population(p_genes)

    # K > N branch (K=15)
    c_genes_gt = np.ones((15, l), dtype=np.float32)
    c_fit_gt, _ = evaluator.evaluate_population(c_genes_gt)
    renders_before = evaluator.renders

    survival = make_exclusive(replacement=make_elite())
    rng = np.random.default_rng(0)
    _ = survival(Population(p_genes, p_fit), Population(c_genes_gt, c_fit_gt), rng)
    assert evaluator.renders == renders_before

    # K <= N branch (K=5)
    c_genes_le = np.ones((5, l), dtype=np.float32)
    c_fit_le, _ = evaluator.evaluate_population(c_genes_le)
    renders_before_le = evaluator.renders
    _ = survival(Population(p_genes, p_fit), Population(c_genes_le, c_fit_le), rng)
    assert evaluator.renders == renders_before_le


# Task 2: Generational Gap G


def test_generational_gap_extremes_zero_and_one():
    """gap=0.0 carries all N from parents; gap=1.0 carries all N from children."""
    n, k, l = 20, 20, 11
    parents = Population(np.ones((n, l), dtype=np.float32) * 1.0, np.full(n, 10.0, dtype=np.float32))
    children = Population(np.ones((k, l), dtype=np.float32) * 2.0, np.full(k, 20.0, dtype=np.float32))
    rng = np.random.default_rng(0)

    # G = 0.0: all parents
    op_0 = make_generational_gap(replacement=make_elite(), gap=0.0)
    surv_0 = op_0(parents, children, rng)
    assert surv_0.genes.shape == (n, l)
    assert np.all(surv_0.genes == 1.0)

    # G = 1.0: all children
    op_1 = make_generational_gap(replacement=make_elite(), gap=1.0)
    surv_1 = op_1(parents, children, rng)
    assert surv_1.genes.shape == (n, l)
    assert np.all(surv_1.genes == 2.0)


def test_generational_gap_split_sum_invariant():
    """Across multiple G and N values, n_prev + n_child always equals N exactly."""
    rng = np.random.default_rng(42)
    l = 11

    for n in (10, 11, 17, 30):
        k = n
        parents = Population(np.ones((n, l), dtype=np.float32) * 1.0, np.arange(n, dtype=np.float32))
        children = Population(np.ones((k, l), dtype=np.float32) * 2.0, np.arange(k, dtype=np.float32))

        for gap in (0.0, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0):
            op = make_generational_gap(replacement=make_elite(), gap=gap)
            survivors = op(parents, children, rng)
            assert survivors.genes.shape == (n, l)
            assert len(survivors.fitness) == n

            n_p = int(np.sum(survivors.genes[:, 0] == 1.0))
            n_c = int(np.sum(survivors.genes[:, 0] == 2.0))
            assert n_p + n_c == n
            assert n_p == int(round((1.0 - gap) * n))


def test_generational_gap_validation():
    """gap outside [0.0, 1.0] raises ConfigError."""
    with pytest.raises(ConfigError, match="generational_gap gap"):
        make_generational_gap(replacement=make_elite(), gap=-0.1)

    with pytest.raises(ConfigError, match="generational_gap gap"):
        make_generational_gap(replacement=make_elite(), gap=1.5)


def test_generational_gap_render_accounting_unchanged():
    """generational_gap survival consumes zero renders."""
    target = np.zeros((16, 16, 3), dtype=np.uint8)
    evaluator = Evaluator(target=target, size=(16, 16))

    n, k, l = 10, 10, 11
    p_genes = np.zeros((n, l), dtype=np.float32)
    p_fit, _ = evaluator.evaluate_population(p_genes)
    c_genes = np.ones((k, l), dtype=np.float32)
    c_fit, _ = evaluator.evaluate_population(c_genes)

    renders_before = evaluator.renders
    op = make_generational_gap(replacement=make_elite(), gap=0.5)
    rng = np.random.default_rng(0)
    _ = op(Population(p_genes, p_fit), Population(c_genes, c_fit), rng)

    assert evaluator.renders == renders_before
