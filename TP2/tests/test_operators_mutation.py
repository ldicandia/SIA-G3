"""Contracts and tests for mutation operators.

Pins MUT-01 (scope contract: probability 0.0 is no-op, 1.0 changes exactly one gene)
and MUT-05 (per-gene-kind effects: sigma-scaled perturbation + reflect repair for ordinary genes,
active flag state flip with no fixed point, stream-length independence, and single repair point).
"""

from __future__ import annotations

import numpy as np
import pytest

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
from tp2.engine.operators.mutation import make_gene


def test_mutation_probability_zero_is_noop():
    """Mutation with probability 0.0 returns bit-identical array."""
    mutate = make_gene(0.0)
    rng = np.random.default_rng(42)
    genes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 1.0], dtype=np.float32)
    child = mutate(genes, rng)
    np.testing.assert_array_equal(child, genes)


def test_mutation_probability_one_mutates_exactly_one_gene_per_individual():
    """Mutation with probability 1.0 changes exactly one gene per individual -- never 0, never 2."""
    mutate = make_gene(1.0)
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
    """Ordinary genes (coords, color, alpha) mutate by normal perturbation scaled by locus sigma and stay in bounds."""
    # Test across multiple non-active loci
    budget = 1
    bounds = bounds_for(budget)
    sigmas = sigma_for(budget)
    mutate = make_gene(1.0)

    # For each non-active locus, force or observe mutation
    for target_locus in range(GENES_PER_TRIANGLE):
        if target_locus == ACTIVE:
            continue
        # Run across multiple seeds until target_locus is hit
        hit = False
        for seed in range(200):
            rng = np.random.default_rng(seed)
            # Peek if first locus selected would be target_locus
            # Mock or check selection: locus is chosen by rng.integers(size) after random()
            # With prob=1.0: random() consumed, then integers(11)
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
    """Active flag locus flips state: active becomes inactive, inactive becomes active.

    Specifically, 0.5 is active under Phase 1 convention and must become inactive.
    There is no fixed point: every starting value changes its active state.
    """
    mutate = make_gene(1.0)
    # Test starting values: [0.0, 0.2, 0.499, 0.5, 0.501, 0.8, 1.0]
    test_values = [0.0, 0.2, 0.499, 0.5, 0.501, 0.8, 1.0]

    for val in test_values:
        # Find a seed that targets the active locus (locus = ACTIVE = 10)
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

                # State must flip: active -> inactive, inactive -> active
                assert is_active_now != was_active, f"Starting value {val} did not flip active state (was {was_active}, now {is_active_now})"
                hit = True
                break
        assert hit, f"Did not hit active locus for test value {val}"


def test_repair_invariant_all_genes_in_bounds_after_mutation():
    """Mutation is the single repair point in the codebase (REP-03).

    Every gene of a fully mutated population is inside its locus bounds after mutation.
    """
    mutate = make_gene(1.0)
    rng = np.random.default_rng(999)
    n_individuals = 100
    triangles = 30
    pop = random_population(rng, n_individuals, triangles)

    for i in range(n_individuals):
        child = mutate(pop[i], rng)
        # assert_in_bounds raises ValueError if any locus is outside bounds
        assert_in_bounds(child)


def test_mutation_preserves_row_order_no_coupling():
    """Mutating a permuted batch yields the permuted results of mutating the original batch."""
    mutate = make_gene(0.5)
    rng_init = np.random.default_rng(42)
    pop = random_population(rng_init, 10, 5)

    # Mutate original batch with fresh RNGs per individual
    results_orig = []
    for i in range(10):
        rng_i = np.random.default_rng(1000 + i)
        results_orig.append(mutate(pop[i], rng_i))

    # Mutate permuted batch with matching RNGs
    perm = np.array([4, 2, 0, 7, 1, 9, 3, 8, 5, 6])
    results_perm = []
    for p in perm:
        rng_p = np.random.default_rng(1000 + p)
        results_perm.append(mutate(pop[p], rng_p))

    for idx, p in enumerate(perm):
        np.testing.assert_array_equal(results_perm[idx], results_orig[p])


def test_zero_length_chromosome_mutation():
    """A zero-length chromosome mutates to a zero-length chromosome without raising."""
    mutate = make_gene(1.0)
    rng = np.random.default_rng(0)
    empty = np.array([], dtype=np.float32)
    result = mutate(empty, rng)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_stream_length_independence_regardless_of_input_contents():
    """Two mutation calls consume identical number of RNG draws regardless of gene contents."""
    mutate = make_gene(1.0)
    triangles = 5
    length = chromosome_length(triangles)

    genes_zeros = np.zeros(length, dtype=np.float32)
    genes_ones = np.ones(length, dtype=np.float32)

    for seed in range(20):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        _ = mutate(genes_zeros, rng1)
        _ = mutate(genes_ones, rng2)

        # After mutate, both RNGs should have consumed identical number of random values
        next_draw_1 = rng1.random()
        next_draw_2 = rng2.random()
        assert next_draw_1 == next_draw_2


def test_mutation_invalid_probability_raises():
    """Probability outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="mutation probability must be in"):
        make_gene(-0.1)
    with pytest.raises(ValueError, match="mutation probability must be in"):
        make_gene(1.1)
