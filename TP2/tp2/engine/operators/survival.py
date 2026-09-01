"""Survival strategies: additive, exclusive, and generational gap.

Provides:
- additive: selects N individuals from union pool of N parents + K children (SUR-01)
- exclusive: if K > N selects N from children; if K <= N takes all K children plus (N - K) parents (SUR-02, SUR-03)
- generational_gap: splits N into round((1-G)*N) parents and round(G*N) children (SUR-05)

All survival operators operate on Population instances without re-evaluating fitness (REP-08).
G (Brecha Generacional) and A/B (selection-blend coefficients) are unrelated parameters
governing different steps of the loop: G governs survival pool split; A/B governs parent selection.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from tp2.engine.genome import Population
from .registry import SURVIVAL

__all__ = [
    "make_additive",
    "make_exclusive",
    "make_generational_gap",
]


@SURVIVAL.register("additive")
def make_additive(replacement: Callable):
    """SUR-01: Additive survival selecting N from N+K union pool."""
    def survive(parents: Population, children: Population, rng: np.random.Generator, ctx: Any = None) -> Population:
        n = parents.genes.shape[0]
        union = Population.concat(parents, children)
        indices = replacement(union.fitness, n, rng, ctx=ctx)
        return Population(union.genes[indices].copy(), union.fitness[indices].copy())
    return survive


@SURVIVAL.register("exclusive")
def make_exclusive(replacement: Callable):
    """SUR-02, SUR-03: Exclusive survival with strict K > N branch boundary.

    Per CATEDRA warning: the branch boundary is K > N, not K >= N.
    At K == N both branches coincide to all K children and 0 parents.
    """
    def survive(parents: Population, children: Population, rng: np.random.Generator, ctx: Any = None) -> Population:
        n = parents.genes.shape[0]
        k = children.genes.shape[0]

        if k > n:
            # Branch K > N: select N from children exclusively
            indices = replacement(children.fitness, n, rng, ctx=ctx)
            return Population(children.genes[indices].copy(), children.fitness[indices].copy())

        # Branch K <= N: all K children unconditionally, plus (N - K) parents
        if k == 0:
            p_indices = replacement(parents.fitness, n, rng, ctx=ctx)
            return Population(parents.genes[p_indices].copy(), parents.fitness[p_indices].copy())

        if k == n:
            return Population(children.genes.copy(), children.fitness.copy())

        # 0 < k < n: children first, then selected parents
        n_needed = n - k
        p_indices = replacement(parents.fitness, n_needed, rng, ctx=ctx)
        sel_parents = Population(parents.genes[p_indices].copy(), parents.fitness[p_indices].copy())
        return Population.concat(Population(children.genes.copy(), children.fitness.copy()), sel_parents)

    return survive


@SURVIVAL.register("generational_gap")
def make_generational_gap(replacement: Callable, gap: float = 0.5):
    """SUR-05: Brecha Generacional G splitting N into (1-G)*N parents and G*N children."""
    from tp2.engine.config import ConfigError

    if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not (0.0 <= gap <= 1.0):
        raise ConfigError(f"generational_gap gap must be in [0.0, 1.0], got {gap!r}")

    gap_val = float(gap)

    def survive(parents: Population, children: Population, rng: np.random.Generator, ctx: Any = None) -> Population:
        n = parents.genes.shape[0]
        # Exact integer split guaranteed to sum to n
        n_prev = int(round((1.0 - gap_val) * n))
        n_child = n - n_prev

        if n_prev == n:
            p_indices = replacement(parents.fitness, n, rng, ctx=ctx)
            return Population(parents.genes[p_indices].copy(), parents.fitness[p_indices].copy())

        if n_child == n:
            c_indices = replacement(children.fitness, n, rng, ctx=ctx)
            return Population(children.genes[c_indices].copy(), children.fitness[c_indices].copy())

        p_indices = replacement(parents.fitness, n_prev, rng, ctx=ctx) if n_prev > 0 else np.array([], dtype=int)
        c_indices = replacement(children.fitness, n_child, rng, ctx=ctx) if n_child > 0 else np.array([], dtype=int)

        p_sel = Population(parents.genes[p_indices].copy(), parents.fitness[p_indices].copy())
        c_sel = Population(children.genes[c_indices].copy(), children.fitness[c_indices].copy())
        return Population.concat(p_sel, c_sel)

    return survive
