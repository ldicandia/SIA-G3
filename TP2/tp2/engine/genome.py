"""Flat chromosome representation and its single repair point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GENES_PER_TRIANGLE = 11
X1, Y1, X2, Y2, X3, Y3, R, G, B, A, ACTIVE = range(GENES_PER_TRIANGLE)
ACTIVE_THRESHOLD = 0.5

BOUNDS_PER_TRIANGLE = np.array(
    [
        [-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1],
        [-0.1, 1.1], [-0.1, 1.1],
        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
        [0.1, 0.9], [0.0, 1.0],
    ],
    dtype=np.float32,
)
SIGMA_PER_TRIANGLE = np.array(
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.08, 0.05, 0.0],
    dtype=np.float32,
)


def chromosome_length(budget: int) -> int:
    """Return the number of alleles in a chromosome with *budget* triangles."""
    if budget < 0:
        raise ValueError(f"triangle budget must be non-negative, got {budget}")
    return GENES_PER_TRIANGLE * budget


def bounds_for(budget: int) -> np.ndarray:
    return np.tile(BOUNDS_PER_TRIANGLE, (budget, 1))


def sigma_for(budget: int) -> np.ndarray:
    return np.tile(SIGMA_PER_TRIANGLE, budget)


def is_active_locus(budget: int) -> np.ndarray:
    mask = np.zeros(chromosome_length(budget), dtype=bool)
    mask[ACTIVE::GENES_PER_TRIANGLE] = True
    return mask


def reflect(v: np.ndarray | float, lo: np.ndarray | float, hi: np.ndarray | float) -> np.ndarray:
    """Reflect values into closed bounds without creating a boundary pile-up.

    Production mutation is its only caller. Reflection, unlike clipping,
    preserves an overshoot's magnitude.
    """
    values = np.asarray(v, dtype=np.float32)
    lower = np.asarray(lo, dtype=np.float32)
    upper = np.asarray(hi, dtype=np.float32)
    span = upper - lower
    if np.any(span <= 0):
        raise ValueError("reflect bounds must have hi > lo")
    t = np.abs(np.mod(values - lower, 2 * span))
    return np.asarray(lower + np.minimum(t, 2 * span - t), dtype=np.float32)


def assert_in_bounds(genes: np.ndarray, bounds: np.ndarray | None = None) -> None:
    """Raise a useful error for the first locus outside its chromosome bounds."""
    flat = np.asarray(genes, dtype=np.float32).reshape(-1)
    expected = bounds_for(flat.size // GENES_PER_TRIANGLE) if bounds is None else bounds
    expected = np.asarray(expected, dtype=np.float32).reshape(-1, 2)
    if flat.size != len(expected):
        raise ValueError(f"genes length {flat.size} does not match {len(expected)} bounds")
    bad = np.flatnonzero((flat < expected[:, 0]) | (flat > expected[:, 1]))
    if bad.size:
        index = int(bad[0])
        raise ValueError(f"gene locus {index}={flat[index]!r} is outside {expected[index].tolist()}")


def random_population(rng: np.random.Generator, n: int, budget: int) -> np.ndarray:
    """Generate an in-bounds C-contiguous float32 population using injected RNG."""
    if n < 0:
        raise ValueError(f"population size must be non-negative, got {n}")
    bounds = bounds_for(budget)
    genes = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, chromosome_length(budget)))
    return np.ascontiguousarray(genes, dtype=np.float32)


def active_mask(genes: np.ndarray) -> np.ndarray:
    return np.asarray(genes).reshape(-1, GENES_PER_TRIANGLE)[:, ACTIVE] >= ACTIVE_THRESHOLD


def active_count(genes: np.ndarray) -> int:
    return int(np.count_nonzero(active_mask(genes)))


@dataclass(slots=True)
class Population:
    """Structure-of-arrays population; fitness stays beside the originating genome."""

    genes: np.ndarray
    fitness: np.ndarray

    def __post_init__(self) -> None:
        if self.genes.ndim != 2:
            raise ValueError("population genes must have shape (N, L)")
        if self.fitness.ndim != 1 or self.fitness.shape[0] != self.genes.shape[0]:
            raise ValueError("population fitness must have shape (N,) matching genes")

    @classmethod
    def concat(cls, first: "Population", second: "Population") -> "Population":
        return cls(
            genes=np.concatenate((first.genes, second.genes), axis=0),
            fitness=np.concatenate((first.fitness, second.fitness), axis=0),
        )
