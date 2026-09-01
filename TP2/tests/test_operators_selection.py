"""Contracts and tests for selection operators.

Pins the cátedra's elite multiplicity formula n(i) = ceil((K - i) / N)
and its behavior on boundary conditions, stability, and degenerate inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from tp2.engine.operators.selection import elite_counts, make_elite, make_random, make_blend


def test_catedra_worked_examples_k4_and_k12():
    """Formula: n(i) = ceil((K - i) / N), where rank i starts from 0.

    Source: CATEDRA.md / Algoritmos Genéticos slide 'Muestreo Directo | Elite'.
    Worked example with N = 7, fitnesses [78, 68, 62, 39, 25, 12, 2]:
      - K = 4  -> counts [1, 1, 1, 1, 0, 0, 0] (best 4 taken once)
      - K = 12 -> counts [2, 2, 2, 2, 2, 1, 1] (best 5 taken twice, others once)
    """
    counts_k4 = elite_counts(4, 7)
    assert list(counts_k4) == [1, 1, 1, 1, 0, 0, 0]

    counts_k12 = elite_counts(12, 7)
    assert list(counts_k12) == [2, 2, 2, 2, 2, 1, 1]

    # Also check through the operator itself
    fitness = np.array([78.0, 68.0, 62.0, 39.0, 25.0, 12.0, 2.0], dtype=np.float64)
    select = make_elite()
    rng = np.random.default_rng(42)

    selected_k4 = select(fitness, 4, rng)
    assert list(selected_k4) == [0, 1, 2, 3]

    selected_k12 = select(fitness, 12, rng)
    # Ranks 0..4 twice, ranks 5..6 once -> [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6]
    assert list(selected_k12) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6]


def test_elite_multiplicity_k_exceeds_n_k10_n4():
    """When K exceeds N, the best individuals are selected more than once."""
    counts = elite_counts(10, 4)
    assert list(counts) == [3, 3, 2, 2]
    assert int(sum(counts)) == 10

    fitness = np.array([10.0, 20.0, 40.0, 30.0], dtype=np.float64)
    # Sorted order of indices by descending fitness: [2, 3, 1, 0]
    select = make_elite()
    rng = np.random.default_rng(0)
    selected = select(fitness, 10, rng)
    # Rank 0 (idx 2) x3, Rank 1 (idx 3) x3, Rank 2 (idx 1) x2, Rank 3 (idx 0) x2
    assert list(selected) == [2, 2, 2, 3, 3, 3, 1, 1, 0, 0]


def test_totality_counts_sum_to_k_sweep():
    """Totality: counts sum to exactly k for all k in [0, 40] across multiple population sizes."""
    for n in (1, 2, 3, 4, 7, 13, 20):
        for k in range(0, 41):
            counts = elite_counts(k, n)
            assert len(counts) == n
            assert int(sum(counts)) == k
            assert all(int(x) >= 0 for x in counts)


def test_boundary_k_equals_n_and_k_equals_n_plus_one():
    """Boundary conditions: k == n gives [1]*n, and k == n + 1 gives [2] + [1]*(n-1)."""
    for n in (1, 2, 3, 4, 7, 13, 25):
        assert list(elite_counts(n, n)) == [1] * n
        assert list(elite_counts(n + 1, n)) == [2] + [1] * (n - 1)


def test_tie_stability_preserves_original_index_order():
    """Individuals with bit-identical fitness are ordered by their original index (stable sort)."""
    fitness = np.array([5.0, 10.0, 10.0, 3.0, 10.0, 1.0], dtype=np.float64)
    # Indices with fitness 10.0 are 1, 2, 4 in that order
    select = make_elite()
    rng = np.random.default_rng(123)
    selected = select(fitness, 3, rng)
    assert list(selected) == [1, 2, 4]

    # Two runs over tied fitness produce the identical index array
    selected_2 = select(fitness, 3, rng)
    np.testing.assert_array_equal(selected, selected_2)


def test_elite_k_zero_returns_empty_array():
    """K of 0 returns an empty integer index array and raises nothing."""
    fitness = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    select = make_elite()
    rng = np.random.default_rng(0)
    selected = select(fitness, 0, rng)
    assert isinstance(selected, np.ndarray)
    assert selected.size == 0
    assert np.issubdtype(selected.dtype, np.integer)


def test_elite_invalid_inputs_raise():
    """Negative K or empty population raises ValueError."""
    select = make_elite()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="selection requires non-empty population"):
        select(np.array([], dtype=np.float64), 5, rng)

    with pytest.raises(ValueError, match="selection requires non-empty population"):
        select(np.array([1.0, 2.0]), -1, rng)
