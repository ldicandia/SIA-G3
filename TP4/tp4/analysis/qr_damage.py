"""Physical deterioration of a +-1 QR matrix: torn corners, stains and the damaged fraction.

Assignment (docs/SIATP42026.pdf page 5, section 2.1): one corner of the table's QR is torn,
and the others are worn, scratched, stained with food or torn.

- A torn corner is missing paper, so its modules read white (-1). The missing region is the
  triangle `distance < size` measured from the corner, where `distance` is the Manhattan
  distance to the corner module (`corner_distance`).
- A stain is a dark (+1) square block at some position.

Only numpy is used. Matrices are square, 2-D and hold -1 and +1; every function returns a copy.
"""

from __future__ import annotations

import numpy as np

CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _check_matrix(matrix) -> np.ndarray:
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 1:
        raise ValueError(f"matrix must be square and 2-D, got shape {m.shape}")
    if not np.isin(m, (-1, 1)).all():
        raise ValueError("matrix values must be -1 or +1")
    return m


def _check_size(size, side: int) -> int:
    if not _is_int(size) or not 1 <= int(size) <= side:
        raise ValueError(f"size must be an integer in [1, {side}], got {size!r}")
    return int(size)


def corner_distance(side: int, corner: str) -> np.ndarray:
    """(side, side) int array: Manhattan distance of every module to the given corner module."""
    if not _is_int(side) or side < 1:
        raise ValueError(f"side must be a positive integer, got {side!r}")
    if corner not in CORNERS:
        raise ValueError(f"unknown corner {corner!r}; allowed: {list(CORNERS)}")
    rows, cols = np.indices((int(side), int(side)))
    r = rows if corner.startswith("top") else side - 1 - rows
    c = cols if corner.endswith("left") else side - 1 - cols
    return (r + c).astype(int)


def torn_corner(matrix, size: int, corner: str) -> np.ndarray:
    """Copy of `matrix` with the modules at corner distance < size set to -1 (white paper)."""
    m = _check_matrix(matrix)
    size = _check_size(size, m.shape[0])
    out = m.astype(np.int8, copy=True)
    out[corner_distance(m.shape[0], corner) < size] = -1
    return out


def occlusion(matrix, size: int, top: int, left: int, value: int = 1) -> np.ndarray:
    """Copy of `matrix` with the size x size block at (top, left) set to `value` (default black)."""
    m = _check_matrix(matrix)
    side = m.shape[0]
    size = _check_size(size, side)
    if value not in (-1, 1):
        raise ValueError(f"value must be -1 or +1, got {value!r}")
    if not (_is_int(top) and _is_int(left)) or not (0 <= top <= side - size and 0 <= left <= side - size):
        raise ValueError(f"a {size}x{size} block at ({top!r}, {left!r}) does not fit in a {side}x{side} matrix")
    out = m.astype(np.int8, copy=True)
    out[int(top) : int(top) + size, int(left) : int(left) + size] = value
    return out


def random_occlusions(matrix, size: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """`n` stained copies of `matrix`: (queries (n, side*side) int8, positions (n, 2) as top, left)."""
    m = _check_matrix(matrix)
    side = m.shape[0]
    size = _check_size(size, side)
    if not _is_int(n) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    positions = rng.integers(0, side - size + 1, size=(int(n), 2))
    queries = np.empty((int(n), side * side), dtype=np.int8)
    for row, (top, left) in enumerate(positions):
        queries[row] = occlusion(m, size, int(top), int(left)).reshape(-1)
    return queries, positions.astype(int)


def changed_fraction(query, original) -> float:
    """Fraction of modules where `query` differs from `original` (same shape required)."""
    q = np.asarray(query)
    o = np.asarray(original)
    if q.shape != o.shape or q.size == 0:
        raise ValueError(f"query and original must have the same non-empty shape, got {q.shape} and {o.shape}")
    return float(np.mean(q != o))
