"""One-point crossover, with a single boundary policy shared by every future crossover.

`locus_mask` is the one place the gene-vs-triangle boundary policy lives: for
the triangle boundary a cut is expressed in triangle units and the mask,
built at that granularity, is expanded to loci with `numpy.repeat`; for the
gene boundary a cut is already expressed in loci and needs no expansion. A
triangle-boundary cut at index `t` therefore builds exactly the mask a
gene-boundary cut at locus `t * genes_per_triangle` builds -- the two
policies agree on the multiples-of-`genes_per_triangle` lattice and differ
everywhere else, which is why comparing them measures the destructiveness of
splitting a triangle rather than measuring noise.

`cut_domain_size` is the seam a future crossover factory (two-point, ring,
uniform) uses: it chooses a cut domain over `cut_domain_size(...)` positions
and builds its own mask over that domain, and inherits both boundary
policies for free without re-validating `boundary` itself. A chromosome
shorter than one triangle cannot happen here -- the triangle-budget range
check (`--triangles`, CLI-side) rejects a budget below one at configuration
time, so `length` is always 0 or a positive multiple of `genes_per_triangle`.
"""

from __future__ import annotations

import numpy as np

from .registry import CROSSOVER

BOUNDARY_CHOICES = ("gene", "triangle")


def _validate_boundary(boundary: str) -> None:
    if boundary not in BOUNDARY_CHOICES:
        from tp2.engine.config import ConfigError
        raise ConfigError(f"crossover boundary must be one of {BOUNDARY_CHOICES}, got {boundary!r}")


def cut_domain_size(length: int, genes_per_triangle: int, boundary: str) -> int:
    """Number of positions a cut may be drawn from under `boundary`."""
    _validate_boundary(boundary)
    return length // genes_per_triangle if boundary == "triangle" else length


def locus_mask(length: int, genes_per_triangle: int, cut: int, boundary: str) -> np.ndarray:
    """Boolean mask of shape (length,), true from `cut` onward in `boundary` units.

    A zero-length chromosome yields an empty mask for either boundary
    policy, raising nothing -- the operator is total.
    """
    domain = cut_domain_size(length, genes_per_triangle, boundary)
    if not 0 <= cut <= domain:
        raise ValueError(f"crossover cut {cut} is outside cut-domain size {domain}")
    domain_mask = np.zeros(domain, dtype=bool)
    domain_mask[cut:] = True
    if boundary == "triangle":
        return np.repeat(domain_mask, genes_per_triangle)
    return domain_mask


def apply_mask(parent_1: np.ndarray, parent_2: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.where(mask, parent_2, parent_1).astype(np.float32), np.where(mask, parent_1, parent_2).astype(np.float32)


@CROSSOVER.register("one_point")
def make_one_point(boundary: str = "gene"):
    _validate_boundary(boundary)

    def crossover(parent_1: np.ndarray, parent_2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        length = parent_1.size
        domain = cut_domain_size(length, 11, boundary)
        cut = int(rng.integers(domain + 1))
        return apply_mask(parent_1, parent_2, locus_mask(length, 11, cut, boundary))
    return crossover
