"""Parent/replacement selection methods from the cátedra.

The `coefficient` used by `blend` below is the selection-blend proportion
`A` -- an approved optional extra -- and is NOT the Brecha Generacional `G`
CATEDRA.md mandates for generational replacement (Phase 3's SUR-05). The two
are orthogonal and deliberately do not share vocabulary; see CATEDRA.md's
"Brecha Generacional G" section and its resolved A/B discrepancy note.
"""

from __future__ import annotations

import numpy as np

from .registry import BLEND_MAX_DEPTH, SELECTION

__all__ = ["elite_counts", "make_elite", "make_random", "make_blend", "BLEND_MAX_DEPTH"]


def elite_counts(k: int, n: int) -> np.ndarray:
    """Course elite multiplicity n(i) = ceil((K-i)/N), by fitness rank."""
    return np.maximum(0, np.ceil((k - np.arange(n)) / n).astype(int))


@SELECTION.register("elite")
def make_elite():
    def select(fitness: np.ndarray, k: int, _: np.random.Generator) -> np.ndarray:
        if k < 0 or fitness.size == 0:
            raise ValueError("selection requires non-empty population and non-negative count")
        ordered = np.argsort(-fitness, kind="stable")
        return np.repeat(ordered, elite_counts(k, fitness.size))
    return select


@SELECTION.register("random")
def make_random():
    """Muestreo Aleatorio (CATEDRA.md): picks at random, with no regard for fitness.

    Gives the blend a second member with genuinely different behaviour --
    Success Criterion 4 is unprovable with only one selection method -- and
    registering it is itself the demonstration that a new operator costs
    zero lines of tp2/engine/loop.py.
    """
    def select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        if k < 0 or fitness.size == 0:
            raise ValueError("selection requires non-empty population and non-negative count")
        return rng.integers(0, fitness.size, size=k)
    return select


@SELECTION.register("blend")
def make_blend(coefficient: float | None = None, method_1: dict | None = None,
               method_2: dict | None = None, depth: int = 0):
    """SEL-09: a coefficient blend of two selection methods.

    Satisfies the same 3-argument protocol its members satisfy -- (fitness,
    k, rng) -> indices -- which is why it nests (T-02-08's depth cap applies
    the same way to a nested blend) and why it works in the replacement slot
    with no special case (SUR-04): survival draws its replacement selection
    from the same SELECTION registry parent selection does.
    """
    from tp2.engine.config import ConfigError

    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not 0.0 <= coefficient <= 1.0:
        raise ConfigError(f"blend coefficient must be a number in [0.0, 1.0], got {coefficient!r}")
    if not isinstance(method_1, dict):
        raise ConfigError(f"blend method_1 must be an operator spec object, got {method_1!r}")
    if not isinstance(method_2, dict):
        raise ConfigError(f"blend method_2 must be an operator spec object, got {method_2!r}")

    select_1 = SELECTION.build(method_1, depth=depth + 1)
    select_2 = SELECTION.build(method_2, depth=depth + 1)

    def select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        # Deterministic split: Python's round() is banker's rounding, so
        # round(0.5 * 11) == 6, not 5. Pinned by a test -- do not "fix" this
        # into a stochastic split later, or seed reproducibility silently
        # breaks.
        n1 = int(round(coefficient * k))
        first = select_1(fitness, n1, rng)
        second = select_2(fitness, k - n1, rng)
        # First method then second: the pairing step downstream consumes
        # this array in the order selection returned it, with no re-sort.
        return np.concatenate([first, second])
    return select
