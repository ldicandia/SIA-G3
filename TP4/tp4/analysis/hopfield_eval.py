"""Evaluation, orthogonality and noise helpers for Hopfield experiments.

Shared by the letter and QR experiments.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..models.hopfield import check_patterns


def n_flips_for(level: float, n: int) -> int:
    """Number of flipped bits for noise `level` on `n` neurons (Python round, half-to-even)."""
    return int(round(float(level) * int(n)))


def flip_bits(
    pattern, n_flips: int, n_queries: int, rng: np.random.Generator
) -> np.ndarray:
    """Create `n_queries` copies of 1-D `pattern`, each flipping `n_flips` distinct positions."""
    pat = np.asarray(pattern, dtype=np.int8).ravel()
    n = pat.size
    k = int(n_flips)
    if k < 0 or k > n:
        raise ValueError(f"n_flips must be in [0, {n}], got {k}")
    if int(n_queries) < 1:
        raise ValueError(f"n_queries must be >= 1, got {n_queries}")
    queries = np.repeat(pat[np.newaxis, :], int(n_queries), axis=0)
    if k > 0:
        for row in range(int(n_queries)):
            pos = rng.choice(n, size=k, replace=False)
            queries[row, pos] = -queries[row, pos]
    return queries


def overlaps(states, patterns) -> np.ndarray:
    """Normalized inner products S @ P^T / N in float64."""
    P = check_patterns(patterns).astype(np.float64)
    s_arr = np.asarray(states, dtype=np.float64)
    n = float(P.shape[1])
    return (s_arr @ P.T) / n


def classify_states(states, patterns) -> tuple[np.ndarray, np.ndarray]:
    """Classify each row of `states` as ('stored', idx), ('inverted', idx), or ('spurious', -1).

    Stored patterns are checked before inverted patterns.
    """
    P = check_patterns(patterns)
    s_arr = np.asarray(states, dtype=np.int8)
    if s_arr.ndim == 1:
        s_arr = s_arr.reshape(1, -1)
    S = check_patterns(s_arr)
    b = S.shape[0]
    classes = np.full(b, "spurious", dtype=object)
    indices = np.full(b, -1, dtype=int)

    for row in range(b):
        st = S[row]
        stored_matches = np.flatnonzero(np.all(P == st, axis=1))
        if stored_matches.size > 0:
            classes[row] = "stored"
            indices[row] = int(stored_matches[0])
            continue
        inv_matches = np.flatnonzero(np.all(-P == st, axis=1))
        if inv_matches.size > 0:
            classes[row] = "inverted"
            indices[row] = int(inv_matches[0])

    return np.asarray(classes, dtype=str), indices


def nearest_pattern(states, patterns) -> np.ndarray:
    """Index of the stored pattern with maximum overlap per state row (ties to lowest index)."""
    s_arr = np.asarray(states, dtype=np.int8)
    if s_arr.ndim == 1:
        s_arr = s_arr.reshape(1, -1)
    ov = overlaps(s_arr, patterns)
    return np.argmax(ov, axis=1).astype(int)


def stability_margins(W, patterns) -> np.ndarray:
    """Per stored pattern, min_i (xi_i * h_i), which is > 0 when every neuron is strictly stable."""
    P = check_patterns(patterns).astype(np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    h = P @ W_arr
    return np.min(P * h, axis=1)


def dot_matrix(patterns) -> np.ndarray:
    """Pairwise integer dot-product matrix P @ P^T."""
    P = check_patterns(patterns).astype(int)
    return P @ P.T


def rank_subsets(
    dots: np.ndarray, k: int
) -> list[tuple[tuple[int, ...], int, float]]:
    """Rank all C(n, k) subsets by (max |off-diagonal dot|, mean |off-diagonal dot|, combo)."""
    dots_arr = np.asarray(dots, dtype=int)
    n = dots_arr.shape[0]
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, {n}], got {k}")
    abs_dots = np.abs(dots_arr)
    items: list[tuple[int, float, tuple[int, ...]]] = []
    for combo in itertools.combinations(range(n), k):
        off_vals = [int(abs_dots[a, b]) for i, a in enumerate(combo) for b in combo[i + 1 :]]
        max_abs = max(off_vals)
        mean_abs = sum(off_vals) / len(off_vals)
        items.append((max_abs, mean_abs, combo))
    items.sort()
    return [(combo, max_abs, mean_abs) for max_abs, mean_abs, combo in items]


def render_rows(pattern, shape: tuple[int, int]) -> list[str]:
    """Render a 1-D {-1, +1} pattern as `shape[0]` strings of '#' (+1) and '.' (-1)."""
    n_rows, n_cols = int(shape[0]), int(shape[1])
    arr = np.asarray(pattern, dtype=np.int8).reshape(n_rows, n_cols)
    return ["".join("#" if int(v) == 1 else "." for v in row) for row in arr]
