"""One-point crossover over either gene or triangle boundaries."""

from __future__ import annotations

import numpy as np

from .registry import CROSSOVER


def locus_mask(length: int, genes_per_triangle: int, cut: int, boundary: str) -> np.ndarray:
    if boundary == "triangle":
        cut *= genes_per_triangle
    if not 0 <= cut <= length:
        raise ValueError(f"crossover cut {cut} is outside chromosome length {length}")
    mask = np.zeros(length, dtype=bool)
    mask[cut:] = True
    return mask


def apply_mask(parent_1: np.ndarray, parent_2: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.where(mask, parent_2, parent_1).astype(np.float32), np.where(mask, parent_1, parent_2).astype(np.float32)


@CROSSOVER.register("one_point")
def make_one_point(boundary: str = "gene"):
    if boundary not in {"gene", "triangle"}:
        raise ValueError(f"crossover boundary must be 'gene' or 'triangle', got {boundary!r}")

    def crossover(parent_1: np.ndarray, parent_2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        length = parent_1.size
        choices = length // 11 + 1 if boundary == "triangle" else length + 1
        cut = int(rng.integers(choices))
        return apply_mask(parent_1, parent_2, locus_mask(length, 11, cut, boundary))
    return crossover
