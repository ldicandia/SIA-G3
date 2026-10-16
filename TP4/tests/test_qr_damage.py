"""Torn-corner, stain and changed-fraction helpers for the QR deterioration experiment."""

from __future__ import annotations

import numpy as np
import pytest

from tp4.analysis.qr_damage import (
    CORNERS,
    changed_fraction,
    corner_distance,
    occlusion,
    random_occlusions,
    torn_corner,
)

SIDE = 11


def _random_matrix(seed: int = 0, side: int = SIDE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random((side, side)) < 0.5, 1, -1).astype(np.int8)


def test_corners_constant_order():
    assert CORNERS == ("top_left", "top_right", "bottom_left", "bottom_right")


def test_corner_distance_is_manhattan_distance_to_each_corner():
    d = corner_distance(5, "top_left")
    assert d.shape == (5, 5)
    assert d[0, 0] == 0 and d[4, 4] == 8 and d[1, 2] == 3
    assert corner_distance(5, "top_right")[0, 4] == 0
    assert corner_distance(5, "bottom_left")[4, 0] == 0
    assert corner_distance(5, "bottom_right")[4, 4] == 0
    assert corner_distance(5, "bottom_right")[0, 0] == 8


def test_torn_corner_of_size_one_changes_at_most_the_corner_module():
    m = _random_matrix(1)
    for corner in CORNERS:
        out = torn_corner(m, 1, corner)
        diff = np.argwhere(out != m)
        assert len(diff) <= 1
        r, c = (0 if "top" in corner else SIDE - 1), (0 if "left" in corner else SIDE - 1)
        assert out[r, c] == -1
        if len(diff):
            assert tuple(diff[0]) == (r, c) and m[r, c] == 1


def test_torn_corner_sets_a_triangle_to_white():
    ones = np.ones((SIDE, SIDE), dtype=np.int8)
    out = torn_corner(ones, SIDE, "top_left")
    assert int(np.sum(out == -1)) == SIDE * (SIDE + 1) // 2
    assert out[0, 0] == -1 and out[SIDE - 1, SIDE - 1] == 1
    small = torn_corner(ones, 3, "top_left")
    assert int(np.sum(small == -1)) == 6
    rr, cc = np.nonzero(small == -1)
    assert np.all(rr + cc < 3)


def test_corner_masks_are_flips_of_the_top_left_mask():
    ones = np.ones((SIDE, SIDE), dtype=np.int8)
    masks = {c: torn_corner(ones, 6, c) == -1 for c in CORNERS}
    assert np.array_equal(masks["top_right"], np.fliplr(masks["top_left"]))
    assert np.array_equal(masks["bottom_left"], np.flipud(masks["top_left"]))
    assert np.array_equal(masks["bottom_right"], np.flipud(np.fliplr(masks["top_left"])))


def test_torn_corner_returns_a_copy_and_leaves_the_rest_untouched():
    m = _random_matrix(2)
    before = m.copy()
    out = torn_corner(m, 4, "bottom_right")
    assert np.array_equal(m, before)
    keep = corner_distance(SIDE, "bottom_right") >= 4
    assert np.array_equal(out[keep], m[keep])
    assert out is not m


def test_occlusion_sets_exactly_the_block_to_black():
    m = -np.ones((SIDE, SIDE), dtype=np.int8)
    out = occlusion(m, 3, 2, 5)
    expected = m.copy()
    expected[2:5, 5:8] = 1
    assert np.array_equal(out, expected)
    assert int(np.sum(out == 1)) == 9
    white = occlusion(np.ones((SIDE, SIDE), dtype=np.int8), 2, 0, 0, value=-1)
    assert int(np.sum(white == -1)) == 4


def test_occlusion_block_must_fit():
    m = _random_matrix(3)
    assert occlusion(m, SIDE, 0, 0).min() == 1
    with pytest.raises(ValueError):
        occlusion(m, 3, SIDE - 2, 0)
    with pytest.raises(ValueError):
        occlusion(m, 3, 0, SIDE - 2)
    with pytest.raises(ValueError):
        occlusion(m, 3, -1, 0)
    with pytest.raises(ValueError):
        occlusion(m, 0, 0, 0)


def test_random_occlusions_stay_inside_the_grid_and_are_deterministic():
    m = _random_matrix(4)
    size = 4
    q1, pos1 = random_occlusions(m, size, 50, np.random.default_rng(7))
    q2, pos2 = random_occlusions(m, size, 50, np.random.default_rng(7))
    assert q1.shape == (50, SIDE * SIDE) and pos1.shape == (50, 2)
    assert np.array_equal(q1, q2) and np.array_equal(pos1, pos2)
    assert pos1.min() >= 0 and pos1.max() <= SIDE - size
    for row, (top, left) in zip(q1, pos1, strict=True):
        assert np.array_equal(row.reshape(SIDE, SIDE), occlusion(m, size, int(top), int(left)))
    q3, _ = random_occlusions(m, size, 50, np.random.default_rng(8))
    assert not np.array_equal(q1, q3)


def test_random_occlusions_full_size_has_one_position():
    m = _random_matrix(5)
    q, pos = random_occlusions(m, SIDE, 3, np.random.default_rng(0))
    assert np.all(pos == 0) and np.all(q == 1)


def test_changed_fraction():
    m = _random_matrix(6)
    assert changed_fraction(m, m) == 0.0
    assert changed_fraction(-m, m) == 1.0
    q = m.copy().reshape(-1)
    q[:12] = -q[:12]
    assert changed_fraction(q, m.reshape(-1)) == pytest.approx(12 / (SIDE * SIDE))
    with pytest.raises(ValueError):
        changed_fraction(m[:5], m)


@pytest.mark.parametrize("size", [0, SIDE + 1, -1])
def test_torn_corner_rejects_a_bad_size(size):
    with pytest.raises(ValueError):
        torn_corner(_random_matrix(), size, "top_left")


def test_torn_corner_rejects_bad_inputs():
    m = _random_matrix()
    with pytest.raises(ValueError):
        torn_corner(m, 3, "middle")
    with pytest.raises(ValueError):
        torn_corner(np.ones((4, 5), dtype=np.int8), 2, "top_left")
    with pytest.raises(ValueError):
        torn_corner(m, 3.0, "top_left")
    with pytest.raises(ValueError):
        torn_corner(m, True, "top_left")
    with pytest.raises(ValueError):
        corner_distance(0, "top_left")
    with pytest.raises(ValueError):
        corner_distance(5, "nowhere")


@pytest.mark.parametrize("size,n", [(0, 3), (SIDE + 1, 3), (3, 0)])
def test_random_occlusions_rejects_bad_arguments(size, n):
    with pytest.raises(ValueError):
        random_occlusions(_random_matrix(), size, n, np.random.default_rng(0))
