"""Contracts and tests for crossover operators.

Pins:
- CRX-01: One-point crossover
- CRX-02: Two-point crossover (half-open [P1, P2), lo==hi empty segment)
- CRX-03: Ring (anular) crossover with modular wrap-around and max length ceil(S/2)
- CRX-04: Uniform crossover (per-locus strict r < p, no positional correlation)
- CRX-05: Shared gene vs triangle cut boundary policy
- Parametrized allele-conservation sweep across all 4 crossovers and 2 boundaries
"""

from __future__ import annotations

import numpy as np
import pytest

from tp2.engine.config import ConfigError
from tp2.engine.operators.crossover import (
    apply_mask,
    cut_domain_size,
    expand_mask,
    locus_mask,
    make_one_point,
    make_ring,
    make_two_point,
    make_uniform,
)
from tp2.engine.operators.registry import CROSSOVER


def test_crossover_registry_contains_all_four_methods():
    """CROSSOVER registry contains one_point, two_point, ring, uniform."""
    names = CROSSOVER.names()
    assert {"one_point", "two_point", "ring", "uniform"} <= set(names)


def test_two_point_exact_mask_controlled_p1_p2():
    """Two-point on domain 10 with P1=3, P2=7 yields True for indices 3..6 and False elsewhere."""
    class MockRNG:
        def __init__(self, p1, p2):
            self.draws = [p1, p2]
            self.idx = 0

        def integers(self, low, high=None):
            val = self.draws[self.idx]
            self.idx += 1
            return val

    # 10 genes
    parent_1 = np.zeros(10, dtype=np.float32)
    parent_2 = np.ones(10, dtype=np.float32)

    op = make_two_point(boundary="gene")
    child_1, child_2 = op(parent_1, parent_2, MockRNG(3, 7))

    # Child 1 gets parent_2 (1.0) where mask is True (indices 3..6)
    expected_c1 = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.float32)
    expected_c2 = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1], dtype=np.float32)
    np.testing.assert_array_equal(child_1, expected_c1)
    np.testing.assert_array_equal(child_2, expected_c2)

    # Order of P1, P2 draw does not matter: P1=7, P2=3 produces identical result
    child_1_rev, child_2_rev = op(parent_1, parent_2, MockRNG(7, 3))
    np.testing.assert_array_equal(child_1_rev, expected_c1)
    np.testing.assert_array_equal(child_2_rev, expected_c2)


def test_two_point_lo_equals_hi_empty_segment():
    """Two-point with P1 == P2 builds an all-false mask (children equal parents)."""
    class MockRNG:
        def integers(self, low, high=None):
            return 4

    parent_1 = np.arange(10, dtype=np.float32)
    parent_2 = np.arange(10, 20, dtype=np.float32)

    op = make_two_point(boundary="gene")
    child_1, child_2 = op(parent_1, parent_2, MockRNG())
    np.testing.assert_array_equal(child_1, parent_1)
    np.testing.assert_array_equal(child_2, parent_2)


def test_two_point_p1_zero_p2_last_minus_one():
    """P1=0 and P2=domain-1 swaps indices 0..domain-2 (leaves last index unswapped)."""
    class MockRNG:
        def __init__(self):
            self.draws = [0, 9]
            self.idx = 0

        def integers(self, low, high=None):
            v = self.draws[self.idx]
            self.idx += 1
            return v

    parent_1 = np.zeros(10, dtype=np.float32)
    parent_2 = np.ones(10, dtype=np.float32)

    op = make_two_point(boundary="gene")
    child_1, child_2 = op(parent_1, parent_2, MockRNG())
    # 0..8 swapped (1.0), index 9 unswapped (0.0)
    expected_c1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.float32)
    np.testing.assert_array_equal(child_1, expected_c1)


def test_uniform_strict_inequality_at_probability_boundary():
    """Uniform crossover uses strict r < p: a draw exactly equal to p is NOT swapped."""
    class MockRNG:
        def random(self, size):
            # 4 loci: [0.49, 0.50, 0.51, 0.0] with p=0.5
            return np.array([0.49, 0.50, 0.51, 0.0])

    parent_1 = np.zeros(4, dtype=np.float32)
    parent_2 = np.ones(4, dtype=np.float32)

    op = make_uniform(p=0.5, boundary="gene")
    child_1, child_2 = op(parent_1, parent_2, MockRNG())

    # Only loci with r < 0.5 (indices 0 and 3) are swapped
    expected_c1 = np.array([1, 0, 0, 1], dtype=np.float32)
    np.testing.assert_array_equal(child_1, expected_c1)


def test_uniform_endpoints_zero_and_one():
    """Uniform with p=0.0 swaps nothing, p=1.0 swaps everything."""
    parent_1 = np.zeros(10, dtype=np.float32)
    parent_2 = np.ones(10, dtype=np.float32)
    rng = np.random.default_rng(42)

    op_0 = make_uniform(p=0.0, boundary="gene")
    c1_0, c2_0 = op_0(parent_1, parent_2, rng)
    np.testing.assert_array_equal(c1_0, parent_1)
    np.testing.assert_array_equal(c2_0, parent_2)

    op_1 = make_uniform(p=1.0, boundary="gene")
    c1_1, c2_1 = op_1(parent_1, parent_2, rng)
    np.testing.assert_array_equal(c1_1, parent_2)
    np.testing.assert_array_equal(c2_1, parent_1)


def test_uniform_positional_independence():
    """Adjacent loci under uniform crossover have statistically independent swap outcomes."""
    op = make_uniform(p=0.5, boundary="gene")
    p1 = np.zeros(20, dtype=np.float32)
    p2 = np.ones(20, dtype=np.float32)
    rng = np.random.default_rng(12345)

    n_trials = 5000
    swaps = np.empty((n_trials, 20), dtype=bool)
    for i in range(n_trials):
        c1, _ = op(p1, p2, rng)
        swaps[i] = (c1 == 1.0)

    # Compute correlation between adjacent loci
    for locus in range(19):
        corr = np.corrcoef(swaps[:, locus].astype(float), swaps[:, locus + 1].astype(float))[0, 1]
        assert abs(corr) < 0.05, f"Loci {locus} and {locus+1} have significant correlation: {corr}"


def test_uniform_validation():
    """Uniform rejects p outside [0.0, 1.0]."""
    with pytest.raises(ConfigError, match="uniform crossover p"):
        make_uniform(p=-0.1)
    with pytest.raises(ConfigError, match="uniform crossover p"):
        make_uniform(p=1.1)


def test_ring_exact_mask_controlled_wrap_around():
    """Ring on domain 10 with start=8, length=3 wraps around to indices 8, 9, 0."""
    class MockRNG:
        def integers(self, low, high=None):
            # First draw start=8, second draw length=3
            if not hasattr(self, "step"):
                self.step = 0
            if self.step == 0:
                self.step += 1
                return 8
            return 3

    parent_1 = np.zeros(10, dtype=np.float32)
    parent_2 = np.ones(10, dtype=np.float32)

    op = make_ring(boundary="gene")
    child_1, child_2 = op(parent_1, parent_2, MockRNG())

    # Indices 8, 9, 0 are 1.0; 1..7 are 0.0
    expected_c1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32)
    np.testing.assert_array_equal(child_1, expected_c1)


def test_ring_length_zero_leaves_parents_untouched():
    """Ring with length=0 produces all-false mask (children equal parents)."""
    class MockRNG:
        def integers(self, low, high=None):
            if not hasattr(self, "step"):
                self.step = 0
            if self.step == 0:
                self.step += 1
                return 4
            return 0  # length = 0

    parent_1 = np.arange(10, dtype=np.float32)
    parent_2 = np.arange(10, 20, dtype=np.float32)

    op = make_ring(boundary="gene")
    c1, c2 = op(parent_1, parent_2, MockRNG())
    np.testing.assert_array_equal(c1, parent_1)
    np.testing.assert_array_equal(c2, parent_2)


def test_ring_drawn_length_never_exceeds_ceil_half():
    """Ring drawn length is in [0, ceil(domain/2)] and never exceeds it."""
    op = make_ring(boundary="gene")
    domain = 11  # ceil(11/2) = 6
    p1 = np.zeros(domain, dtype=np.float32)
    p2 = np.ones(domain, dtype=np.float32)
    rng = np.random.default_rng(999)

    max_observed_swapped = 0
    for _ in range(500):
        c1, _ = op(p1, p2, rng)
        swapped = int(np.sum(c1 == 1.0))
        max_observed_swapped = max(max_observed_swapped, swapped)

    assert max_observed_swapped <= 6, f"Ring swapped {max_observed_swapped} loci, exceeding ceil(11/2) = 6"


def test_ring_single_triangle_domain():
    """Ring on domain of size 1 (1 triangle under triangle boundary) builds without raising."""
    p1 = np.zeros(11, dtype=np.float32)
    p2 = np.ones(11, dtype=np.float32)
    op = make_ring(boundary="triangle")
    rng = np.random.default_rng(0)

    for _ in range(20):
        c1, c2 = op(p1, p2, rng)
        assert c1.size == 11 and c2.size == 11
        # Either all 0s or all 1s
        assert np.all(c1 == 0.0) or np.all(c1 == 1.0)


@pytest.mark.parametrize("crossover_name", ["one_point", "two_point", "ring", "uniform"])
@pytest.mark.parametrize("boundary", ["gene", "triangle"])
def test_crossover_allele_conservation_determinism_and_empty(crossover_name: str, boundary: str):
    """Parametrized sweep over all 4 crossovers and 2 boundaries:

    - Allele conservation: child1 + child2 == parent1 + parent2 elementwise
    - Determinism: same seed produces identical children
    - Zero-length chromosome: returns empty arrays without raising
    """
    spec = {"method": crossover_name, "boundary": boundary}
    if crossover_name == "uniform":
        spec["p"] = 0.5

    op = CROSSOVER.build(spec)

    # 1. Zero-length chromosome
    empty1 = np.array([], dtype=np.float32)
    empty2 = np.array([], dtype=np.float32)
    c1_e, c2_e = op(empty1, empty2, np.random.default_rng(0))
    assert c1_e.size == 0 and c2_e.size == 0

    # 2. Allele conservation across multiple seeds
    triangles = 6
    length = triangles * 11
    p1 = np.random.default_rng(1).uniform(-0.1, 1.1, size=length).astype(np.float32)
    p2 = np.random.default_rng(2).uniform(-0.1, 1.1, size=length).astype(np.float32)

    for seed in (10, 20, 30, 40, 50):
        rng_a = np.random.default_rng(seed)
        c1_a, c2_a = op(p1, p2, rng_a)

        # Allele conservation: elementwise sum of children equals sum of parents
        np.testing.assert_allclose(c1_a + c2_a, p1 + p2, atol=1e-6)

        # Each locus in c1 must come from either p1 or p2
        from_p1 = np.isclose(c1_a, p1)
        from_p2 = np.isclose(c1_a, p2)
        assert np.all(from_p1 | from_p2)

        # Determinism: rerun with same seed
        rng_b = np.random.default_rng(seed)
        c1_b, c2_b = op(p1, p2, rng_b)
        np.testing.assert_array_equal(c1_a, c1_b)
        np.testing.assert_array_equal(c2_a, c2_b)
