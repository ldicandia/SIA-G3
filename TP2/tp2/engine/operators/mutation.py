"""Per-gene mutation; repair happens only here."""

from __future__ import annotations

import numpy as np

from tp2.engine.genome import ACTIVE, GENES_PER_TRIANGLE, bounds_for, reflect, sigma_for
from .registry import MUTATION


@MUTATION.register("gene")
def make_gene(probability: float):
    if not 0 <= probability <= 1:
        raise ValueError(f"mutation probability must be in [0, 1], got {probability}")

    def mutate(genes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        child = np.asarray(genes, dtype=np.float32).copy()
        if rng.random() >= probability:
            return child
        locus = int(rng.integers(child.size))
        if locus % GENES_PER_TRIANGLE == ACTIVE:
            child[locus] = 0.0 if child[locus] >= 0.5 else 1.0
            return child
        bounds = bounds_for(child.size // GENES_PER_TRIANGLE)
        child[locus] += rng.normal(0.0, sigma_for(child.size // GENES_PER_TRIANGLE)[locus])
        child[locus:locus + 1] = reflect(child[locus:locus + 1], bounds[locus:locus + 1, 0], bounds[locus:locus + 1, 1])
        return child
    return mutate
