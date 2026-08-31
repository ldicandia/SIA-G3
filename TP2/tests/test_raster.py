from __future__ import annotations

import numpy as np

from tp2.engine.genome import ACTIVE, GENES_PER_TRIANGLE, random_population
from tp2.engine.raster import BACKGROUND, render
from tests._reference import reference_render


def test_raster_is_bit_exact_against_independent_over_compositing_oracle() -> None:
    genes = random_population(np.random.default_rng(20260829), 1, 12)[0]
    actual = render(genes, (64, 64))
    expected = reference_render(genes, (64, 64))
    assert np.array_equal(actual, expected), (np.abs(actual.astype(int) - expected.astype(int)).max(), np.count_nonzero(actual != expected))


def test_inactive_triangle_is_a_true_no_op() -> None:
    genes = random_population(np.random.default_rng(2), 1, 2)[0]
    genes[ACTIVE] = 0.49
    deleted = genes[GENES_PER_TRIANGLE:]
    assert np.array_equal(render(genes, (32, 32)), render(deleted, (32, 32)))


def test_empty_and_all_inactive_genomes_are_background() -> None:
    expected = np.full((32, 32, 3), BACKGROUND, dtype=np.uint8)
    assert np.array_equal(render(np.array([], dtype=np.float32), (32, 32)), expected)
    genes = random_population(np.random.default_rng(1), 1, 3)[0]
    genes[ACTIVE::GENES_PER_TRIANGLE] = 0.49
    assert np.array_equal(render(genes, (32, 32)), expected)


def test_chromosome_order_changes_overlapping_translucent_triangles() -> None:
    genes = np.array([
        0, 0, 1, 0, 0.5, 1, 1, 0, 0, .5, 1,
        0, 0, 1, 0, 0.5, 1, 0, 0, 1, .5, 1,
    ], dtype=np.float32)
    swapped = genes.reshape(-1, 11)[::-1].reshape(-1)
    assert not np.array_equal(render(genes, (64, 64)), render(swapped, (64, 64)))
