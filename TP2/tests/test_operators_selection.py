"""Contracts and tests for selection operators.

Pins the cátedra's selection methods:
- Elite multiplicity formula n(i) = ceil((K - i) / N)
- Roulette and Universal (SUS) over relative aptitude
- Ranking pseudo-fitness f'(i) = (N - rank(i)) / N (worst individual is never selected)
- Boltzmann selection with exponential temperature schedule and overflow-safe ExpVal
- Deterministic tournament with M distinct candidates
- Probabilistic tournament with Threshold and strict less-fit fallback
"""

from __future__ import annotations

import numpy as np
import pytest

from tp2.engine.config import ConfigError
from tp2.engine.operators.registry import SELECTION
from tp2.engine.operators.selection import (
    boltzmann_exp_val,
    elite_counts,
    make_blend,
    make_boltzmann,
    make_elite,
    make_random,
    make_ranking,
    make_roulette,
    make_tournament_deterministic,
    make_tournament_probabilistic,
    make_universal,
    ranking_pseudo_fitness,
)


def test_selection_registry_contains_all_nine_methods():
    """SELECTION registry contains all 9 mandated selection methods."""
    names = SELECTION.names()
    mandated = {
        "elite",
        "random",
        "blend",
        "roulette",
        "universal",
        "ranking",
        "boltzmann",
        "tournament_deterministic",
        "tournament_probabilistic",
    }
    assert mandated <= set(names), f"Missing selection methods: {mandated - set(names)}"


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
    assert list(selected_k12) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6]


def test_elite_multiplicity_k_exceeds_n_k10_n4():
    """When K exceeds N, the best individuals are selected more than once."""
    counts = elite_counts(10, 4)
    assert list(counts) == [3, 3, 2, 2]
    assert int(sum(counts)) == 10

    fitness = np.array([10.0, 20.0, 40.0, 30.0], dtype=np.float64)
    select = make_elite()
    rng = np.random.default_rng(0)
    selected = select(fitness, 10, rng)
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
    select = make_elite()
    rng = np.random.default_rng(123)
    selected = select(fitness, 3, rng)
    assert list(selected) == [1, 2, 4]

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


def test_roulette_selection():
    """Roulette selection draws k indices proportional to fitness."""
    fitness = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    select = make_roulette()
    rng = np.random.default_rng(42)

    indices = select(fitness, 20_000, rng)
    counts = np.bincount(indices, minlength=4)
    freqs = counts / 20_000
    expected = fitness / np.sum(fitness)  # [0.1, 0.2, 0.3, 0.4]
    np.testing.assert_allclose(freqs, expected, atol=0.015)


def test_universal_selection_determinism_and_single_draw():
    """Universal selection is deterministic and consumes exactly one RNG draw."""
    fitness = np.array([78.0, 68.0, 62.0, 39.0, 25.0, 12.0, 2.0], dtype=np.float64)
    select = make_universal()

    # Same seed -> identical output
    res1 = select(fitness, 10, np.random.default_rng(100))
    res2 = select(fitness, 10, np.random.default_rng(100))
    np.testing.assert_array_equal(res1, res2)

    # Exactly one RNG draw consumed
    rng1 = np.random.default_rng(200)
    rng2 = np.random.default_rng(200)
    _ = select(fitness, 15, rng1)
    _ = rng2.random()
    assert rng1.bit_generator.state == rng2.bit_generator.state


def test_ranking_pseudo_fitness_formula_and_worst_never_selected():
    """Ranking assigns f'(i) = (N - rank(i)) / N and never samples worst individual (f'=0)."""
    # Cátedra N=7 worked example
    fitness = np.array([78.0, 68.0, 62.0, 39.0, 25.0, 12.0, 2.0], dtype=np.float64)
    f_prime = ranking_pseudo_fitness(fitness)

    # Sorted order is already 0..6, so f_prime = [6, 5, 4, 3, 2, 1, 0] / 7
    expected_f_prime = np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.float64) / 7.0
    np.testing.assert_allclose(f_prime, expected_f_prime)

    # Worst individual has f' == 0.0
    assert f_prime[6] == 0.0

    # Over 20,000 draws, rank N (index 6) must NEVER be sampled
    select = make_ranking()
    rng = np.random.default_rng(42)
    sampled = select(fitness, 20_000, rng)
    assert 6 not in sampled, "Worst individual (rank N, f'=0) was sampled by ranking selection!"


def test_ranking_tie_stability():
    """Ranking preserves original-index order for tied fitness values (stable sort)."""
    fitness = np.array([10.0, 20.0, 20.0, 5.0], dtype=np.float64)
    # Ranks: idx 1 is rank 1, idx 2 is rank 2 (tie broken by original index), idx 0 is rank 3, idx 3 is rank 4
    f_prime = ranking_pseudo_fitness(fitness)
    # (4 - [3, 1, 2, 4]) / 4 = [1/4, 3/4, 2/4, 0/4] = [0.25, 0.75, 0.5, 0.0]
    expected = np.array([0.25, 0.75, 0.5, 0.0])
    np.testing.assert_allclose(f_prime, expected)


def test_boltzmann_mean_expval_is_one_and_handles_low_temperature():
    """Boltzmann ExpVal has mean ~1.0 across configurations and produces no NaN/inf even at T=0.001."""
    fitness = np.array([0.1, 0.5, 0.8, 0.95], dtype=np.float64)

    test_cases = [
        # (t0, tc, k, gen)
        (10.0, 1.0, 0.01, 0),       # High T ~ 10.0
        (2.0, 0.1, 0.05, 50),       # Mid T
        (0.01, 0.001, 0.1, 100),    # Very low T ~ 0.001
    ]

    for t0, tc, k, gen in test_cases:
        exp_val = boltzmann_exp_val(fitness, t0, tc, k, generation=gen)
        assert np.isfinite(exp_val).all(), f"NaN or Inf in exp_val for t0={t0}, tc={tc}, gen={gen}"
        assert not np.isnan(exp_val).any()
        np.testing.assert_allclose(
            np.mean(exp_val), 1.0, atol=1e-5,
            err_msg=f"Mean of ExpVal is not ~1.0 for t0={t0}, tc={tc}, gen={gen}",
        )


def test_boltzmann_selection_sampling():
    """Boltzmann selection samples from expected value distribution."""
    fitness = np.array([0.2, 0.4, 0.8, 0.9], dtype=np.float64)
    select = make_boltzmann(t0=5.0, tc=0.5, k=0.01)
    rng = np.random.default_rng(42)

    indices = select(fitness, 10_000, rng)
    assert len(indices) == 10_000
    # Highest fitness (idx 3) sampled most
    counts = np.bincount(indices, minlength=4)
    assert counts[3] > counts[0]


def test_boltzmann_config_validation():
    """Boltzmann rejects invalid parameters at build time."""
    with pytest.raises(ConfigError, match="boltzmann t0"):
        make_boltzmann(t0=-1.0, tc=1.0)

    with pytest.raises(ConfigError, match="boltzmann t0"):
        make_boltzmann(t0=0.0, tc=1.0)

    with pytest.raises(ConfigError, match="boltzmann tc"):
        make_boltzmann(t0=1.0, tc=0.0)

    with pytest.raises(ConfigError, match="boltzmann tc"):
        make_boltzmann(t0=1.0, tc=-0.5)

    with pytest.raises(ConfigError, match="boltzmann k"):
        make_boltzmann(t0=1.0, tc=0.5, k=-0.1)


def test_tournament_deterministic_scaling():
    """Deterministic tournament scales from uniform random (M=1) to elite (M=N)."""
    fitness = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

    # M = 1 degenerates to uniform random selection
    t1 = make_tournament_deterministic(m=1)
    rng = np.random.default_rng(123)
    samples_m1 = t1(fitness, 20_000, rng)
    counts_m1 = np.bincount(samples_m1, minlength=4)
    # Each index roughly 25%
    np.testing.assert_allclose(counts_m1 / 20_000, [0.25, 0.25, 0.25, 0.25], atol=0.02)

    # M = N always returns the fittest index (idx 3)
    t_n = make_tournament_deterministic(m=4)
    samples_mn = t_n(fitness, 100, rng)
    assert (samples_mn == 3).all()

    # M > N is clipped to N and returns fittest without raising
    t_big = make_tournament_deterministic(m=100)
    samples_big = t_big(fitness, 50, rng)
    assert (samples_big == 3).all()


def test_tournament_deterministic_config_validation():
    """Deterministic tournament rejects m < 1."""
    with pytest.raises(ConfigError, match="tournament_deterministic m"):
        make_tournament_deterministic(m=0)
    with pytest.raises(ConfigError, match="tournament_deterministic m"):
        make_tournament_deterministic(m=-2)


def test_tournament_probabilistic_threshold_branches_and_boundary():
    """Probabilistic tournament picks fitter when r < Threshold and LESS FIT when r >= Threshold."""
    fitness = np.array([10.0, 50.0], dtype=np.float64)  # idx 0 less fit, idx 1 fitter

    t_prob = make_tournament_probabilistic(threshold=0.75)

    # Controlled RNG class to test strict inequality boundary
    class MockRNG:
        def __init__(self, r_val):
            self.r_val = r_val

        def choice(self, n, size, replace):
            return np.array([0, 1])

        def random(self):
            return self.r_val

    # r < threshold (0.70 < 0.75) -> picks fitter (idx 1)
    res_fitter = t_prob(fitness, 1, MockRNG(0.70))
    assert res_fitter[0] == 1

    # r == threshold (0.75 == 0.75) -> strict inequality fails, picks LESS FIT (idx 0)
    res_boundary = t_prob(fitness, 1, MockRNG(0.75))
    assert res_boundary[0] == 0, "Strict r < Threshold contract violated: r == Threshold must pick less fit!"

    # r > threshold (0.80 > 0.75) -> picks LESS FIT (idx 0)
    res_less_fit = t_prob(fitness, 1, MockRNG(0.80))
    assert res_less_fit[0] == 0

    # Threshold = 1.0 always returns fitter
    t_100 = make_tournament_probabilistic(threshold=1.0)
    rng = np.random.default_rng(42)
    samples_100 = t_100(fitness, 1000, rng)
    assert (samples_100 == 1).all()

    # Threshold = 0.5 gives ~50/50 between fitter and less fit
    t_50 = make_tournament_probabilistic(threshold=0.5)
    samples_50 = t_50(fitness, 20_000, rng)
    counts_50 = np.bincount(samples_50, minlength=2)
    np.testing.assert_allclose(counts_50 / 20_000, [0.5, 0.5], atol=0.02)


def test_tournament_probabilistic_config_validation():
    """Probabilistic tournament rejects threshold outside [0.5, 1.0]."""
    with pytest.raises(ConfigError, match="tournament_probabilistic threshold"):
        make_tournament_probabilistic(threshold=0.49)
    with pytest.raises(ConfigError, match="tournament_probabilistic threshold"):
        make_tournament_probabilistic(threshold=1.01)
