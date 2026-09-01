"""Crossover operators with shared gene-vs-triangle cut boundary policy.

`cut_domain_size` is the seam every crossover factory (one_point, two_point,
ring, uniform) uses: it operates over `cut_domain_size(...)` units (triangles
or genes) and builds its own mask over that domain, inheriting the boundary
expansion policy for free without re-validating `boundary` itself.
"""

from __future__ import annotations

import numpy as np

from .registry import CROSSOVER

BOUNDARY_CHOICES = ("gene", "triangle")

__all__ = [
    "cut_domain_size",
    "locus_mask",
    "expand_mask",
    "apply_mask",
    "make_one_point",
    "make_two_point",
    "make_ring",
    "make_uniform",
]


def _validate_boundary(boundary: str) -> None:
    if boundary not in BOUNDARY_CHOICES:
        from tp2.engine.config import ConfigError
        raise ConfigError(f"crossover boundary must be one of {BOUNDARY_CHOICES}, got {boundary!r}")


def cut_domain_size(length: int, genes_per_triangle: int, boundary: str) -> int:
    """Number of positions a cut may be drawn from under `boundary`."""
    _validate_boundary(boundary)
    return length // genes_per_triangle if boundary == "triangle" else length


def expand_mask(domain_mask: np.ndarray, genes_per_triangle: int, boundary: str) -> np.ndarray:
    """Expand a boolean mask from cut-domain granularity to individual loci."""
    if boundary == "triangle":
        return np.repeat(domain_mask, genes_per_triangle)
    return domain_mask


def locus_mask(length: int, genes_per_triangle: int, cut: int, boundary: str) -> np.ndarray:
    """Boolean mask of shape (length,), true from `cut` onward in `boundary` units."""
    domain = cut_domain_size(length, genes_per_triangle, boundary)
    if not 0 <= cut <= domain:
        raise ValueError(f"crossover cut {cut} is outside cut-domain size {domain}")
    domain_mask = np.zeros(domain, dtype=bool)
    domain_mask[cut:] = True
    return expand_mask(domain_mask, genes_per_triangle, boundary)


def apply_mask(parent_1: np.ndarray, parent_2: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Swap parent alleles where mask is True, returning two child genomes."""
    return np.where(mask, parent_2, parent_1).astype(np.float32), np.where(mask, parent_1, parent_2).astype(np.float32)


@CROSSOVER.register("one_point")
def make_one_point(boundary: str = "gene"):
    """CRX-01: One-point crossover."""
    _validate_boundary(boundary)

    def crossover(parent_1: np.ndarray, parent_2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        length = parent_1.size
        domain = cut_domain_size(length, 11, boundary)
        if domain == 0:
            return parent_1.copy(), parent_2.copy()
        cut = int(rng.integers(domain + 1))
        return apply_mask(parent_1, parent_2, locus_mask(length, 11, cut, boundary))
    return crossover


@CROSSOVER.register("two_point")
def make_two_point(boundary: str = "gene"):
    """CRX-02: Two-point crossover with random P1, P2 in [0, S-1]."""
    _validate_boundary(boundary)

    def crossover(parent_1: np.ndarray, parent_2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        length = parent_1.size
        domain = cut_domain_size(length, 11, boundary)
        if domain == 0:
            return parent_1.copy(), parent_2.copy()
        p1 = int(rng.integers(0, domain))
        p2 = int(rng.integers(0, domain))
        lo, hi = min(p1, p2), max(p1, p2)
        idx = np.arange(domain)
        # lo == hi produces an all-false mask (empty segment)
        domain_mask = (idx >= lo) & (idx < hi)
        mask = expand_mask(domain_mask, 11, boundary)
        return apply_mask(parent_1, parent_2, mask)
    return crossover


@CROSSOVER.register("ring")
def make_ring(boundary: str = "gene"):
    """CRX-03: Ring (anular) crossover with random P in [0, S-1] and L in [0, ceil(S/2)]."""
    _validate_boundary(boundary)

    def crossover(parent_1: np.ndarray, parent_2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        length = parent_1.size
        domain = cut_domain_size(length, 11, boundary)
        if domain == 0:
            return parent_1.copy(), parent_2.copy()
        max_l = -(-domain // 2)  # Exact integer ceil(domain / 2)
        start = int(rng.integers(0, domain))
        l_len = int(rng.integers(0, max_l + 1))
        if l_len == 0:
            domain_mask = np.zeros(domain, dtype=bool)
        else:
            idx = np.arange(domain)
            domain_mask = ((idx - start) % domain) < l_len
        mask = expand_mask(domain_mask, 11, boundary)
        return apply_mask(parent_1, parent_2, mask)
    return crossover


@CROSSOVER.register("uniform")
def make_uniform(p: float = 0.5, boundary: str = "gene"):
    """CRX-04: Uniform crossover swapping each locus independently with probability p."""
    _validate_boundary(boundary)
    from tp2.engine.config import ConfigError

    if isinstance(p, bool) or not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
        raise ConfigError(f"uniform crossover p must be in [0.0, 1.0], got {p!r}")

    p_val = float(p)

    def crossover(parent_1: np.ndarray, parent_2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        length = parent_1.size
        domain = cut_domain_size(length, 11, boundary)
        if domain == 0:
            return parent_1.copy(), parent_2.copy()
        # Strict r < p swap; no positional correlation between adjacent units
        r = rng.random(domain)
        domain_mask = r < p_val
        mask = expand_mask(domain_mask, 11, boundary)
        return apply_mask(parent_1, parent_2, mask)
    return crossover
