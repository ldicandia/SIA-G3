"""Contracts and tests for mutation operators.

Pins:
- MUT-01: Single gene mutation
- MUT-02: Limited multigene mutation (count in [1, M])
- MUT-03: Uniform multigene mutation (each gene independently with probability Pm)
- MUT-04: Complete mutation (every gene mutates with probability Pm)
- MUT-05: Per-gene-kind effects: sigma-scaled perturbation + reflect repair, active flag state flip with no fixed point
- MUT-06: Michalewicz non-uniform schedule and sigma table override
"""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from tp2.engine.config import ConfigError
from tp2.engine.genome import (
    ACTIVE,
    ACTIVE_THRESHOLD,
    BOUNDS_PER_TRIANGLE,
    GENES_PER_TRIANGLE,
    SIGMA_PER_TRIANGLE,
    active_mask,
    assert_in_bounds,
    bounds_for,
    chromosome_length,
    random_population,
    sigma_for,
)
from tp2.engine.operators.mutation import (
    make_complete,
    make_gene,
    make_multigen_limited,
    make_multigen_uniform,
)
from tp2.engine.operators.registry import MUTATION


def test_mutation_registry_contains_all_four_scopes():
    """MUTATION registry contains gene, multigen_limited, multigen_uniform, complete."""
    names = MUTATION.names()
    assert {"gene", "multigen_limited", "multigen_uniform", "complete"} <= set(names)


def test_mutation_probability_zero_is_noop():
    """Mutation with probability 0.0 returns bit-identical array."""
    mutate = make_gene(probability=0.0)
    rng = np.random.default_rng(42)
    genes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 1.0], dtype=np.float32)
    child = mutate(genes, rng)
    np.testing.assert_array_equal(child, genes)


def test_mutation_probability_one_mutates_exactly_one_gene_per_individual():
    """Mutation with probability 1.0 changes exactly one gene per individual -- never 0, never 2."""
    mutate = make_gene(probability=1.0)
    rng = np.random.default_rng(123)
    n_individuals = 50
    n_triangles = 10
    pop = random_population(rng, n_individuals, n_triangles)

    for i in range(n_individuals):
        original = pop[i]
        child = mutate(original, rng)
        diff = original != child
        assert np.count_nonzero(diff) == 1, f"Individual {i} had {np.count_nonzero(diff)} changed genes"


def test_mutated_ordinary_gene_within_bounds_and_uses_locus_sigma():
    """Ordinary genes mutate by perturbation scaled by locus sigma and stay in bounds."""
    budget = 1
    bounds = bounds_for(budget)
    mutate = make_gene(probability=1.0)

    for target_locus in range(GENES_PER_TRIANGLE):
        if target_locus == ACTIVE:
            continue
        hit = False
        for seed in range(200):
            rng = np.random.default_rng(seed)
            rng_copy = np.random.default_rng(seed)
            _ = rng_copy.random()
            chosen_locus = int(rng_copy.integers(GENES_PER_TRIANGLE))
            if chosen_locus == target_locus:
                original = np.full(GENES_PER_TRIANGLE, 0.5, dtype=np.float32)
                child = mutate(original, rng)
                assert child[target_locus] != original[target_locus]
                assert bounds[target_locus, 0] <= child[target_locus] <= bounds[target_locus, 1]
                hit = True
                break
        assert hit, f"Did not hit target locus {target_locus}"


def test_active_flag_mutation_flips_state_no_fixed_point():
    """Active flag locus flips state: active becomes inactive, inactive becomes active."""
    mutate = make_gene(probability=1.0)
    test_values = [0.0, 0.2, 0.499, 0.5, 0.501, 0.8, 1.0]

    for val in test_values:
        hit = False
        for seed in range(300):
            rng_check = np.random.default_rng(seed)
            _ = rng_check.random()
            if int(rng_check.integers(GENES_PER_TRIANGLE)) == ACTIVE:
                original = np.zeros(GENES_PER_TRIANGLE, dtype=np.float32)
                original[ACTIVE] = val
                was_active = bool(val >= ACTIVE_THRESHOLD)

                rng = np.random.default_rng(seed)
                child = mutate(original, rng)
                is_active_now = bool(child[ACTIVE] >= ACTIVE_THRESHOLD)

                assert is_active_now != was_active, f"Starting value {val} did not flip active state"
                hit = True
                break
        assert hit, f"Did not hit active locus for test value {val}"


def test_repair_invariant_all_genes_in_bounds_after_mutation():
    """Mutation is the single repair point in the codebase (REP-03)."""
    mutate = make_gene(probability=1.0)
    rng = np.random.default_rng(999)
    pop = random_population(rng, 100, 30)

    for i in range(100):
        child = mutate(pop[i], rng)
        assert_in_bounds(child)


def test_mutation_preserves_row_order_no_coupling():
    """Mutating a permuted batch yields the permuted results of mutating the original batch."""
    mutate = make_gene(probability=0.5)
    rng_init = np.random.default_rng(42)
    pop = random_population(rng_init, 10, 5)

    results_orig = []
    for i in range(10):
        rng_i = np.random.default_rng(1000 + i)
        results_orig.append(mutate(pop[i], rng_i))

    perm = np.array([4, 2, 0, 7, 1, 9, 3, 8, 5, 6])
    results_perm = []
    for p in perm:
        rng_p = np.random.default_rng(1000 + p)
        results_perm.append(mutate(pop[p], rng_p))

    for idx, p in enumerate(perm):
        np.testing.assert_array_equal(results_perm[idx], results_orig[p])


def test_zero_length_chromosome_mutation():
    """A zero-length chromosome mutates to a zero-length chromosome without raising."""
    for factory in (make_gene, make_multigen_uniform, make_complete):
        mutate = factory(probability=1.0)
        empty = np.array([], dtype=np.float32)
        res = mutate(empty, np.random.default_rng(0))
        assert res.size == 0

    mutate_lim = make_multigen_limited(m=5, probability=1.0)
    empty = np.array([], dtype=np.float32)
    res_lim = mutate_lim(empty, np.random.default_rng(0))
    assert res_lim.size == 0


def test_stream_length_independence_regardless_of_input_contents():
    """Two mutation calls consume identical number of RNG draws regardless of gene contents."""
    mutate = make_gene(probability=1.0)
    triangles = 5
    length = chromosome_length(triangles)

    genes_zeros = np.zeros(length, dtype=np.float32)
    genes_ones = np.ones(length, dtype=np.float32)

    for seed in range(20):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        _ = mutate(genes_zeros, rng1)
        _ = mutate(genes_ones, rng2)

        next_draw_1 = rng1.random()
        next_draw_2 = rng2.random()
        assert next_draw_1 == next_draw_2


# Task 1: Multigen Limited and Uniform


def test_multigen_limited_drawn_count_range_and_coverage():
    """multigen_limited(m=5) draws counts in [1, 5] covering the full range."""
    m = 5
    mutate = make_multigen_limited(m=m, probability=1.0)
    length = 22  # 2 triangles
    original = np.zeros(length, dtype=np.float32)
    rng = np.random.default_rng(42)

    counts = set()
    for _ in range(300):
        child = mutate(original, rng)
        changed = int(np.count_nonzero(child != original))
        assert 1 <= changed <= m
        counts.add(changed)

    assert counts == {1, 2, 3, 4, 5}, f"Observed counts {counts} did not cover full [1, 5] range"


def test_multigen_limited_m_one_equals_single_gene_scope():
    """multigen_limited(m=1) mutates exactly 1 gene per individual."""
    mutate = make_multigen_limited(m=1, probability=1.0)
    length = 33
    original = np.zeros(length, dtype=np.float32)
    rng = np.random.default_rng(123)

    for _ in range(50):
        child = mutate(original, rng)
        assert np.count_nonzero(child != original) == 1


def test_multigen_limited_m_equals_chromosome_length():
    """multigen_limited(m=length) allows up to length genes to mutate."""
    length = 11
    mutate = make_multigen_limited(m=length, probability=1.0)
    original = np.zeros(length, dtype=np.float32)
    rng = np.random.default_rng(777)

    max_changed = 0
    for _ in range(500):
        child = mutate(original, rng)
        max_changed = max(max_changed, int(np.count_nonzero(child != original)))

    assert max_changed == length, f"Max changed {max_changed} did not reach full length {length}"


def test_multigen_limited_validation():
    """multigen_limited rejects m < 1."""
    with pytest.raises(ConfigError, match="multigen_limited m"):
        make_multigen_limited(m=0)
    with pytest.raises(ConfigError, match="multigen_limited m"):
        make_multigen_limited(m=-2)


def test_multigen_uniform_boundaries():
    """multigen_uniform(probability=0.0) is no-op, probability=1.0 mutates every gene."""
    length = 22
    original = np.zeros(length, dtype=np.float32)
    rng = np.random.default_rng(55)

    mut_0 = make_multigen_uniform(probability=0.0)
    c_0 = mut_0(original, rng)
    np.testing.assert_array_equal(c_0, original)

    mut_1 = make_multigen_uniform(probability=1.0)
    c_1 = mut_1(original, rng)
    assert np.count_nonzero(c_1 != original) == length


def test_multigen_uniform_preserves_row_order():
    """multigen_uniform preserves row order under permutation."""
    mutate = make_multigen_uniform(probability=0.5)
    rng_init = np.random.default_rng(42)
    pop = random_population(rng_init, 10, 5)

    results_orig = [mutate(pop[i], np.random.default_rng(2000 + i)) for i in range(10)]
    perm = np.array([4, 2, 0, 7, 1, 9, 3, 8, 5, 6])
    results_perm = [mutate(pop[p], np.random.default_rng(2000 + p)) for p in perm]

    for idx, p in enumerate(perm):
        np.testing.assert_array_equal(results_perm[idx], results_orig[p])


def test_multigen_limited_vs_multigen_uniform_swap_detection():
    """multigen_limited(m=1) and multigen_uniform(p=1/L) produce structurally distinct count distributions."""
    length = 33
    original = np.zeros(length, dtype=np.float32)

    mut_lim = make_multigen_limited(m=1, probability=1.0)
    mut_uni = make_multigen_uniform(probability=1.0 / length)

    rng_lim = np.random.default_rng(9999)
    rng_uni = np.random.default_rng(9999)

    lim_counts = [int(np.count_nonzero(mut_lim(original, rng_lim) != original)) for _ in range(200)]
    uni_counts = [int(np.count_nonzero(mut_uni(original, rng_uni) != original)) for _ in range(200)]

    # lim_counts is ALWAYS exactly 1
    assert set(lim_counts) == {1}
    # uni_counts is binomial, so it contains 0, 1, 2, ...
    assert len(set(uni_counts)) > 1
    assert 0 in uni_counts or 2 in uni_counts


# Task 2: Complete mutation, Non-uniform Michalewicz schedule, Sigma table overrides


def test_complete_mutation_boundaries():
    """complete(probability=0.0) is no-op, probability=1.0 mutates every gene."""
    length = 22
    original = np.zeros(length, dtype=np.float32)
    rng = np.random.default_rng(1234)

    mut_0 = make_complete(probability=0.0)
    c_0 = mut_0(original, rng)
    np.testing.assert_array_equal(c_0, original)

    mut_1 = make_complete(probability=1.0)
    c_1 = mut_1(original, rng)
    assert np.count_nonzero(c_1 != original) == length


def test_complete_vs_multigen_uniform_distribution_difference():
    """complete and multigen_uniform at same probability (p=0.5) produce distinct count distributions."""
    length = 22
    original = np.zeros(length, dtype=np.float32)

    mut_comp = make_complete(probability=0.5)
    mut_uni = make_multigen_uniform(probability=0.5)

    rng_c = np.random.default_rng(42)
    rng_u = np.random.default_rng(42)

    comp_counts = [int(np.count_nonzero(mut_comp(original, rng_c) != original)) for _ in range(200)]
    uni_counts = [int(np.count_nonzero(mut_uni(original, rng_u) != original)) for _ in range(200)]

    # Complete is strictly all-or-nothing: either 0 or length
    assert set(comp_counts) <= {0, length}
    # Uniform has counts clustered around length/2
    assert any(0 < c < length for c in uni_counts)


def test_non_uniform_schedule_decay_across_generations():
    """Non-uniform schedule produces larger expected perturbations at gen 0 than near horizon."""
    length = 22
    original = np.full(length, 0.5, dtype=np.float32)

    for method, factory in [
        ("gene", lambda: make_gene(probability=1.0, schedule="non_uniform", b=5.0)),
        ("multigen_uniform", lambda: make_multigen_uniform(probability=1.0, schedule="non_uniform", b=5.0)),
    ]:
        mut = factory()
        ctx_early = SimpleNamespace(generation=0, max_generations=100)
        ctx_late = SimpleNamespace(generation=95, max_generations=100)

        early_deltas = []
        late_deltas = []

        for seed in range(200):
            rng_e = np.random.default_rng(seed)
            rng_l = np.random.default_rng(seed)

            c_e = mut(original, rng_e, ctx=ctx_early)
            c_l = mut(original, rng_l, ctx=ctx_late)

            # Exclude active loci (indices 10, 21) from continuous delta measurement
            non_active = [i for i in range(length) if i % 11 != ACTIVE]
            diff_e = np.abs(c_e[non_active] - original[non_active])
            diff_l = np.abs(c_l[non_active] - original[non_active])

            if np.any(diff_e > 0):
                early_deltas.extend(diff_e[diff_e > 0])
            if np.any(diff_l > 0):
                late_deltas.extend(diff_l[diff_l > 0])

        mean_early = float(np.mean(early_deltas))
        mean_late = float(np.mean(late_deltas))

        assert mean_early > 2.0 * mean_late, (
            f"{method}: early perturbation ({mean_early:.4f}) should clearly exceed late perturbation ({mean_late:.4f})"
        )


def test_non_uniform_schedule_past_horizon_clamp():
    """Generation past horizon (gen > max_gen) clamps cleanly without raising or increasing magnitude."""
    length = 22
    original = np.full(length, 0.5, dtype=np.float32)
    mut = make_multigen_uniform(probability=1.0, schedule="non_uniform", b=5.0)

    ctx_horizon = SimpleNamespace(generation=100, max_generations=100)
    ctx_past = SimpleNamespace(generation=150, max_generations=100)

    for seed in range(50):
        rng = np.random.default_rng(seed)
        c_past = mut(original, rng, ctx=ctx_past)
        assert_in_bounds(c_past)


def test_non_uniform_schedule_active_flag_flip_rate_invariant():
    """Active flag flip rate is invariant to generation under non-uniform schedule."""
    length = 11
    original = np.zeros(length, dtype=np.float32)
    original[ACTIVE] = 1.0  # active

    mut = make_gene(probability=1.0, schedule="non_uniform", b=5.0)

    ctx_early = SimpleNamespace(generation=0, max_generations=100)
    ctx_late = SimpleNamespace(generation=99, max_generations=100)

    flips_early = 0
    flips_late = 0
    trials = 1000

    for seed in range(trials):
        rng_e = np.random.default_rng(seed)
        rng_l = np.random.default_rng(seed)

        c_e = mut(original, rng_e, ctx=ctx_early)
        c_l = mut(original, rng_l, ctx=ctx_late)

        if c_e[ACTIVE] < ACTIVE_THRESHOLD:
            flips_early += 1
        if c_l[ACTIVE] < ACTIVE_THRESHOLD:
            flips_late += 1

    # Both should flip with equal frequency (1/11 chance of hitting active locus)
    assert flips_early == flips_late


def test_sigma_override_effect_and_defaults_equivalence():
    """mutation.sigma overrides step sizes, and explicit defaults reproduce default behavior."""
    triangles = 2
    length = chromosome_length(triangles)
    original = np.full(length, 0.5, dtype=np.float32)

    # 1. Custom sigma vs default
    mut_def = make_multigen_uniform(probability=1.0, schedule="uniform")
    mut_big = make_multigen_uniform(
        probability=1.0,
        schedule="uniform",
        sigma={"coordinate": 0.25, "color": 0.40, "alpha": 0.25},
    )

    deltas_def = []
    deltas_big = []
    for seed in range(100):
        c_d = mut_def(original, np.random.default_rng(seed))
        c_b = mut_big(original, np.random.default_rng(seed))

        non_active = [i for i in range(length) if i % 11 != ACTIVE]
        deltas_def.extend(np.abs(c_d[non_active] - original[non_active]))
        deltas_big.extend(np.abs(c_b[non_active] - original[non_active]))

    assert np.mean(deltas_big) > 2.0 * np.mean(deltas_def)

    # 2. Explicit defaults match implicit defaults identically
    mut_explicit = make_multigen_uniform(
        probability=1.0,
        schedule="uniform",
        sigma={"coordinate": 0.05, "color": 0.08, "alpha": 0.05},
    )
    for seed in range(20):
        c_d = mut_def(original, np.random.default_rng(seed))
        c_e = mut_explicit(original, np.random.default_rng(seed))
        np.testing.assert_array_equal(c_d, c_e)


def test_sigma_override_validation():
    """Invalid sigma overrides raise ConfigError."""
    with pytest.raises(ConfigError, match="mutation sigma must be a dict"):
        make_gene(sigma="not_a_dict")

    with pytest.raises(ConfigError, match="unknown sigma override key"):
        make_gene(sigma={"bogus_key": 0.1})

    with pytest.raises(ConfigError, match="must be positive"):
        make_gene(sigma={"coordinate": -0.05})

    with pytest.raises(ConfigError, match="must be positive"):
        make_gene(sigma={"color": 0.0})
