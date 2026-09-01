"""Unit tests for tp2/experiments/aggregate.py's pure aggregation math.

All fixtures are hand-written, tiny, synthetic metrics.csv files under
tmp_path -- never a dependency on any real (gitignored) run directory --
matching this project's own testing convention (tiny, isolated, deterministic
fixtures rather than coupling tests to runtime artifacts that may not exist
in a fresh clone or on a collaborator's machine).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from tp2.experiments.aggregate import align_on_grid, load_seed_curves, median_iqr


def _write_metrics_csv(path: Path, renders: list[float], best_fitness: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["generation", "renders", "best_fitness", "diversity"])
        writer.writeheader()
        for i, (r, f) in enumerate(zip(renders, best_fitness)):
            writer.writerow({"generation": i, "renders": r, "best_fitness": f, "diversity": 1.0 / (i + 1)})


def test_load_seed_curves_orders_by_directory_name_not_readdir_order(tmp_path):
    cell_dir = tmp_path / "cell"
    # Written to disk in reverse order, to prove the returned order is not
    # filesystem readdir order.
    _write_metrics_csv(cell_dir / "seed1" / "metrics.csv", [10, 20], [0.5, 0.6])
    _write_metrics_csv(cell_dir / "seed0" / "metrics.csv", [10, 20], [0.1, 0.2])

    curves = load_seed_curves(cell_dir)

    assert len(curves) == 2
    assert curves[0][1][0] == pytest.approx(0.1)  # seed0 first
    assert curves[1][1][0] == pytest.approx(0.5)  # seed1 second


def test_load_seed_curves_reads_the_requested_x_and_y_columns(tmp_path):
    cell_dir = tmp_path / "cell"
    _write_metrics_csv(cell_dir / "seed0" / "metrics.csv", [10, 20, 30], [0.1, 0.2, 0.3])

    x, y = load_seed_curves(cell_dir)[0]

    assert list(x) == pytest.approx([10.0, 20.0, 30.0])
    assert list(y) == pytest.approx([0.1, 0.2, 0.3])


def test_load_seed_curves_can_read_a_different_y_column(tmp_path):
    cell_dir = tmp_path / "cell"
    _write_metrics_csv(cell_dir / "seed0" / "metrics.csv", [10, 20], [0.1, 0.2])

    _, y = load_seed_curves(cell_dir, y_col="diversity")[0]

    assert list(y) == pytest.approx([1.0, 0.5])


def test_align_on_grid_caps_at_the_minimum_per_seed_maximum_render_count():
    curves = [
        (np.array([0.0, 25.0, 50.0]), np.array([0.1, 0.2, 0.3])),
        (np.array([0.0, 40.0, 80.0]), np.array([0.1, 0.25, 0.4])),
    ]

    grid_x, values = align_on_grid(curves, num_points=10)

    assert grid_x[-1] == pytest.approx(50.0)
    assert grid_x[0] == pytest.approx(0.0)
    assert values.shape == (2, 10)


def test_align_on_grid_resamples_via_last_observation_carried_forward_not_interpolation():
    # A curve that jumps from 0.0 to 10.0 exactly at renders=10. LOCF at
    # renders=5 must still read 0.0 (the last OBSERVED value at or before 5);
    # linear interpolation would report 5.0, a value never actually observed
    # by any individual (best_fitness is piecewise-constant between evals).
    curves = [(np.array([0.0, 10.0]), np.array([0.0, 10.0]))]

    grid_x, values = align_on_grid(curves, num_points=11)  # grid: 0,1,2,...,10

    five_idx = list(grid_x).index(5.0)
    assert values[0, five_idx] == pytest.approx(0.0)
    assert values[0, -1] == pytest.approx(10.0)


def test_median_iqr_single_curve_returns_zero_width_band_not_nan():
    values = np.array([[0.1, 0.2, 0.3]])

    median, q1, q3 = median_iqr(values)

    assert list(median) == pytest.approx([0.1, 0.2, 0.3])
    assert list(q1) == pytest.approx(list(median))
    assert list(q3) == pytest.approx(list(median))
    assert not np.isnan(median).any()


def test_median_iqr_multi_seed_computes_median_and_iqr_band():
    values = np.array(
        [
            [0.1, 0.5],
            [0.2, 0.6],
            [0.9, 0.7],
        ]
    )

    median, q1, q3 = median_iqr(values)

    assert median[0] == pytest.approx(0.2)
    assert q1[0] <= median[0] <= q3[0]
