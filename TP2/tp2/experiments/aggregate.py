"""Pure render-count-grid alignment and median/IQR math across seeds.

No `matplotlib` import anywhere in this file -- kept separate from
`scripts/generate_plots.py` so this module's aggregation math is
independently unit-testable without any display-backend concern. Every
figure `scripts/generate_plots.py` builds is required to go through
`align_on_grid` + `median_iqr` from here, never a bespoke per-figure
aggregation, so the render-count-alignment methodology is identical across
every comparative figure (T-04-10's "degrade explicitly, never silently
fabricate a spread" mitigation lives in `median_iqr` below).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

__all__ = ["load_seed_curves", "align_on_grid", "median_iqr"]


def _read_curve(metrics_path: Path, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    with Path(metrics_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        xs: list[float] = []
        ys: list[float] = []
        for row in reader:
            xs.append(float(row[x_col]))
            ys.append(float(row[y_col]))
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def load_seed_curves(
    cell_dir: str | Path, x_col: str = "renders", y_col: str = "best_fitness"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read every `seed*/metrics.csv` under `cell_dir`, in a fixed order.

    Ordered by `sorted(cell_dir.glob("seed*"))` -- a lexicographic sort of
    the directory NAME (`seed0` < `seed1` < ... < `seed10`) -- never by
    filesystem readdir order. This is the ONE place seed order is ever
    determined for aggregation.

    Note this sorts `seed10` before `seed2` lexicographically. `04-03`'s
    shipped `main_matrix.json` never exceeds single-digit replicate indices
    (`seeds: 5` -> `seed0`..`seed4`), so this is a documented, acceptable
    simplification; a future spec with 10+ seeds would need zero-padded
    directory names or a numeric-aware sort key.
    """
    cell_path = Path(cell_dir)
    return [_read_curve(seed_dir / "metrics.csv", x_col, y_col) for seed_dir in sorted(cell_path.glob("seed*"))]


def align_on_grid(
    curves: list[tuple[np.ndarray, np.ndarray]], num_points: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Resample every curve onto one shared render-count grid via LOCF.

    `cap` is the MINIMUM of each curve's own final (maximum) render count,
    never the maximum across curves -- the boundary resolution: no seed's
    curve is ever read past the render count it actually reached.

    Resampling is last-observation-carried-forward (a step function),
    DELIBERATELY not `numpy.interp`: `best_fitness` is piecewise-constant
    between generations (a value that does not change until the next
    evaluation), so linearly interpolating between two generations' points
    would report a fitness value that was never actually observed by any
    individual.
    """
    cap = min(x[-1] for x, _y in curves)
    grid_x = np.linspace(0, cap, num_points)
    resampled = []
    for x, y in curves:
        idx = np.clip(np.searchsorted(x, grid_x, side="right") - 1, 0, len(x) - 1)
        resampled.append(y[idx])
    values = np.stack(resampled, axis=0)
    return grid_x, values


def median_iqr(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median and interquartile band across seeds (axis 0 = seed).

    For a `values` array with only one row (a single-seed cell),
    `numpy.median`/`numpy.percentile` over a length-1 axis naturally return
    that one row for all three outputs -- NumPy's own median/percentile
    definitions already produce the degenerate zero-width band this
    project's own truth requires (a single-seed cell's aggregation degrades
    to a zero-width IQR band around its one curve, never NaN or an
    exception). No special-casing is needed here.
    """
    median = np.median(values, axis=0)
    q1, q3 = np.percentile(values, [25, 75], axis=0)
    return median, q1, q3
