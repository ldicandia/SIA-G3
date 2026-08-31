from __future__ import annotations

import numpy as np
import pytest

from tp2.engine.genome import ACTIVE, GENES_PER_TRIANGLE, active_count, active_mask, assert_in_bounds, reflect, random_population


def test_random_population_shape_dtype_and_bounds() -> None:
    genes = random_population(np.random.default_rng(7), 8, 30)
    assert genes.shape == (8, 330)
    assert genes.dtype == np.float32
    assert_in_bounds(genes[0])


def test_active_threshold_and_count() -> None:
    genes = np.zeros(GENES_PER_TRIANGLE * 2, dtype=np.float32)
    genes[ACTIVE] = 0.5
    genes[GENES_PER_TRIANGLE + ACTIVE] = 0.4999
    assert active_mask(genes).tolist() == [True, False]
    assert active_count(genes) == 1


def test_reflect_preserves_bounds_and_repairs_outliers() -> None:
    values = np.array([-10, -0.1, 0, 1.1, 31], dtype=np.float32)
    repaired = reflect(values, -0.1, 1.1)
    assert repaired.dtype == np.float32
    assert repaired[1:4].tolist() == values[1:4].tolist()
    assert np.all((repaired >= -0.1) & (repaired <= 1.1))
    assert reflect(np.array([], dtype=np.float32), 0, 1).shape == (0,)


def test_out_of_bounds_reports_locus() -> None:
    genes = random_population(np.random.default_rng(3), 1, 1)[0]
    genes[0] = 2
    with pytest.raises(ValueError, match="locus 0"):
        assert_in_bounds(genes)
