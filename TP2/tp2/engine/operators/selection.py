"""Parent/replacement selection methods from the cátedra."""

from __future__ import annotations

import numpy as np

from .registry import SELECTION


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
