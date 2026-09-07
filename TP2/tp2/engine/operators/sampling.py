"""Shared cumulative-wheel sampling engine for selection operators.

Drives Roulette, Universal (SUS), Ranking, and Boltzmann selections over a
cumulative weight wheel.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sample_from_weights", "scale_fitness", "SCALING_MODES"]

SCALING_MODES = ("none", "offset", "sigma")


def scale_fitness(fitness: np.ndarray, mode: str = "none", sigma_c: float = 2.0) -> np.ndarray:
    """Shift a fitness vector so its spread, not its offset, drives the wheel.

    `fitness = 1 - RMSE` sits on a large constant offset (a blank canvas
    already scores 0.695), and proportional selection divides by the sum, so
    that offset absorbs nearly all the probability mass: measured
    p_best/p_mean is 1.097 on a real generation-0 population. Subtracting a
    baseline restores the spread as the signal.

    Only meaningful for the raw-fitness wheels (roulette, SUS). It is a
    mathematical no-op for Boltzmann, whose exp(f/T) turns any additive shift
    into a constant factor that cancels against the ExpVal mean -- verified
    bit-exact at T = 5.0, 0.5 and 0.002. Never wire it there.
    """
    f = np.asarray(fitness, dtype=np.float64)
    if mode == "none":
        return f
    if mode == "offset":
        scaled = f - f.min()
    elif mode == "sigma":
        scaled = np.maximum(f - (f.mean() - sigma_c * f.std()), 0.0)
    else:
        raise ValueError(f"unknown scaling mode {mode!r}, expected one of {SCALING_MODES}")

    # Every subtraction above zeroes at least the worst individual, and a
    # uniform population (or a lone one) collapses the whole vector to zero,
    # which sample_from_weights rejects outright. Fall back to uniform so a
    # converged population stays selectable -- the same "must stay
    # selectable" guarantee ranking_pseudo_fitness makes for n == 1.
    if scaled.sum() <= 0:
        return np.ones_like(f)
    return scaled


def sample_from_weights(
    weights: np.ndarray | list[float],
    k: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample k indices proportional to positive weights using roulette or SUS.

    Parameters
    ----------
    weights : array-like
        Non-negative weight vector for each candidate.
    k : int
        Number of items to sample.
    mode : {"roulette", "sus"}
        Sampling strategy:
        - "roulette": k independent draws ~ Uniform[0, 1).
        - "sus": single draw r ~ Uniform[0, 1), stratified into r_j = (r + j)/k.
    rng : np.random.Generator
        Injected random number generator.

    Returns
    -------
    np.ndarray
        Integer array of shape (k,) containing sampled indices in [0, len(weights)).
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        raise ValueError("weights vector cannot be empty")
    if np.any(w < 0):
        raise ValueError(f"weights must be non-negative, found negative values: {w[w < 0]}")
    total = np.sum(w)
    if total <= 0:
        raise ValueError(f"sum of weights must be strictly positive, got {total}")

    if k == 0:
        return np.empty(0, dtype=int)

    # Cumulative distribution q_i in float64
    q = np.cumsum(w) / total

    if mode == "roulette":
        # k independent draws
        r_j = rng.random(k)
    elif mode == "sus":
        # Single draw stratified across k intervals
        r = rng.random()
        r_j = (r + np.arange(k, dtype=np.float64)) / k
    else:
        raise ValueError(f"unknown sampling mode {mode!r}, expected 'roulette' or 'sus'")

    # searchsorted with side='left' finds index i such that q_{i-1} < r_j <= q_i
    indices = np.searchsorted(q, r_j, side="left")
    # Clip to guard against floating-point boundary at 1.0
    return np.clip(indices, 0, len(w) - 1).astype(int)
