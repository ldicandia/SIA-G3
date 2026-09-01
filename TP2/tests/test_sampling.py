"""Tests for the shared cumulative-wheel sampling engine (sample_from_weights)."""

from __future__ import annotations

import numpy as np
import pytest

from tp2.engine.operators.sampling import sample_from_weights


def test_roulette_empirical_frequencies():
    """Roulette mode samples indices proportional to weights over large sample count."""
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    rng = np.random.default_rng(42)
    k = 40_000
    indices = sample_from_weights(weights, k, mode="roulette", rng=rng)

    counts = np.bincount(indices, minlength=len(weights))
    freqs = counts / k
    expected_freqs = weights / np.sum(weights)  # [0.1, 0.2, 0.3, 0.4]

    # Generous tolerance (0.015) for 40,000 draws
    np.testing.assert_allclose(freqs, expected_freqs, atol=0.015)


def test_sus_single_draw_generator_state():
    """Universal (SUS) consumes exactly ONE random number from the generator regardless of k."""
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    for k in (1, 5, 50, 500):
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)

        _ = sample_from_weights(weights, k, mode="sus", rng=rng1)

        # rng2 makes exactly one random draw
        _ = rng2.random()

        # Both RNGs should now be in identical internal state
        assert rng1.bit_generator.state == rng2.bit_generator.state


def test_sus_stratified_distribution():
    """Universal (SUS) samples proportionally with low variance across stratified intervals."""
    weights = np.array([2.0, 2.0, 2.0, 2.0])  # Equal weights
    rng = np.random.default_rng(999)
    # k = 4 with equal weights must select each index exactly once
    indices = sample_from_weights(weights, 4, mode="sus", rng=rng)
    assert sorted(indices.tolist()) == [0, 1, 2, 3]


def test_validation_errors():
    """Raises ValueError on invalid weights or parameters."""
    rng = np.random.default_rng(0)

    # Negative k
    with pytest.raises(ValueError, match="k must be non-negative"):
        sample_from_weights([1.0, 2.0], -1, "roulette", rng)

    # Empty weights
    with pytest.raises(ValueError, match="weights vector cannot be empty"):
        sample_from_weights([], 5, "roulette", rng)

    # Negative weights
    with pytest.raises(ValueError, match="weights must be non-negative"):
        sample_from_weights([1.0, -0.5, 2.0], 5, "roulette", rng)

    # All-zero weights
    with pytest.raises(ValueError, match="sum of weights must be strictly positive"):
        sample_from_weights([0.0, 0.0, 0.0], 5, "roulette", rng)

    # Unknown mode
    with pytest.raises(ValueError, match="unknown sampling mode"):
        sample_from_weights([1.0, 2.0], 5, "invalid_mode", rng)


def test_k_zero_returns_empty_array():
    """k=0 returns empty int array without error."""
    rng = np.random.default_rng(0)
    res = sample_from_weights([1.0, 2.0, 3.0], 0, "roulette", rng)
    assert isinstance(res, np.ndarray)
    assert res.size == 0
    assert np.issubdtype(res.dtype, np.integer)
