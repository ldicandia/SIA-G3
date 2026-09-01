"""Diversity metrics and structure stagnation analysis.

Provides scale-free population diversity and rank-aligned whole-genome
stability comparison for stop conditions and metric reporting.
"""

from __future__ import annotations

import numpy as np


def diversity(genes: np.ndarray, bounds: np.ndarray) -> float:
    """Scale-free population diversity.

    Computes the mean over loci of (stdev / locus_span). Locus span division
    ensures coordinate loci (span 1.2), RGB loci (span 1.0), and alpha loci
    (span 0.8) are properly normalized without over-weighting coordinates.
    """
    if genes.size == 0 or len(genes) <= 1:
        return 0.0
    stdev = genes.std(axis=0)
    span = bounds[:, 1] - bounds[:, 0]
    # Avoid division by zero if a locus has zero span
    safe_span = np.where(span > 0, span, 1.0)
    div = stdev / safe_span
    return float(np.mean(div))


def unchanged_fraction(
    prev_genes: np.ndarray,
    prev_fitness: np.ndarray,
    curr_genes: np.ndarray,
    curr_fitness: np.ndarray,
    tolerance: float,
) -> float:
    """Fraction of population unchanged between consecutive generations.

    Compares individuals aligned by descending fitness rank (stable sort).
    Whole-genome mean absolute distance is compared against `tolerance` with
    a strict <=.

    Limitation: rank alignment can compare distinct individuals under near-tie
    churn, but under selection pressure top fitness ranks remain stable.
    """
    if prev_genes.size == 0 or curr_genes.size == 0:
        return 1.0

    prev_order = np.argsort(-prev_fitness, kind="stable")
    curr_order = np.argsort(-curr_fitness, kind="stable")

    prev_ranked = prev_genes[prev_order]
    curr_ranked = curr_genes[curr_order]

    # Mean absolute distance per individual across all loci
    dist = np.abs(prev_ranked - curr_ranked).mean(axis=1)
    return float((dist <= tolerance).mean())
