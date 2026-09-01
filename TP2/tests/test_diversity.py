"""Contracts and tests for diversity and genome stability."""

from __future__ import annotations

import numpy as np
import pytest

from tp2.engine.diversity import diversity, unchanged_fraction
from tp2.engine.genome import bounds_for, chromosome_length, random_population


def test_diversity_zero_for_identical_population():
    """A population with zero variance across all loci has diversity 0.0."""
    triangles = 5
    bounds = bounds_for(triangles)
    length = chromosome_length(triangles)

    # 20 identical individuals
    genes = np.tile(np.full(length, 0.5, dtype=np.float32), (20, 1))
    div = diversity(genes, bounds)
    assert div == 0.0


def test_diversity_finite_and_positive_for_random_population():
    """A random population has positive, finite scale-free diversity."""
    rng = np.random.default_rng(42)
    triangles = 10
    bounds = bounds_for(triangles)
    pop = random_population(rng, 50, triangles)

    div = diversity(pop, bounds)
    assert np.isfinite(div)
    assert div > 0.1  # Clearly positive and normalized


def test_diversity_empty_or_single_individual():
    """Diversity on <=1 individual returns 0.0 without raising."""
    bounds = bounds_for(2)
    assert diversity(np.array([]), bounds) == 0.0
    single = np.full((1, chromosome_length(2)), 0.5, dtype=np.float32)
    assert diversity(single, bounds) == 0.0


def test_unchanged_fraction_identical_and_fully_moved():
    """unchanged_fraction returns 1.0 when identical and 0.0 when all exceed tolerance."""
    n = 20
    length = 22
    prev_genes = np.full((n, length), 0.5, dtype=np.float32)
    prev_fitness = np.linspace(0.1, 0.9, n, dtype=np.float32)

    # 1. Identical copies
    curr_genes = prev_genes.copy()
    curr_fitness = prev_fitness.copy()
    assert unchanged_fraction(prev_genes, prev_fitness, curr_genes, curr_fitness, tolerance=0.01) == 1.0

    # 2. Shift all genes by 0.1 (exceeding tolerance 0.01)
    moved_genes = prev_genes + 0.1
    assert unchanged_fraction(prev_genes, prev_fitness, moved_genes, curr_fitness, tolerance=0.01) == 0.0


def test_unchanged_fraction_permutation_invariance():
    """unchanged_fraction is invariant to identically permuting both generations."""
    rng = np.random.default_rng(123)
    n = 15
    triangles = 3
    length = chromosome_length(triangles)

    prev_genes = rng.uniform(0.0, 1.0, size=(n, length)).astype(np.float32)
    prev_fitness = rng.uniform(0.1, 0.9, size=n).astype(np.float32)

    curr_genes = prev_genes + rng.normal(0.0, 0.005, size=(n, length)).astype(np.float32)
    curr_fitness = prev_fitness + rng.normal(0.0, 0.01, size=n).astype(np.float32)

    frac_orig = unchanged_fraction(prev_genes, prev_fitness, curr_genes, curr_fitness, tolerance=0.01)

    # Permute both generations
    perm = rng.permutation(n)
    frac_perm = unchanged_fraction(
        prev_genes[perm],
        prev_fitness[perm],
        curr_genes[perm],
        curr_fitness[perm],
        tolerance=0.01,
    )

    assert frac_orig == frac_perm
