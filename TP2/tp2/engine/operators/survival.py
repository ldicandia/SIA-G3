"""Additive survival retains fitness with its genome."""

from __future__ import annotations

import numpy as np

from tp2.engine.genome import Population
from .registry import SURVIVAL


@SURVIVAL.register("additive")
def make_additive(replacement):
    def survive(parents: Population, children: Population, rng: np.random.Generator) -> Population:
        union = Population.concat(parents, children)
        indices = replacement(union.fitness, parents.genes.shape[0], rng)
        return Population(union.genes[indices].copy(), union.fitness[indices].copy())
    return survive
