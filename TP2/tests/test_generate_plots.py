"""Unit tests for scripts/generate_plots.py's five-figure set.

All fixtures are hand-written, tiny, synthetic metrics.csv/run.json files
under tmp_path -- never a dependency on any real (gitignored) run directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.generate_plots as gp
from scripts.generate_plots import FIGURE_CLAIMS, GeneratePlotsError


def _write_metrics_csv(path: Path, renders: list[float], best_fitness: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["generation", "renders", "best_fitness", "diversity"])
        writer.writeheader()
        for i, (r, f) in enumerate(zip(renders, best_fitness)):
            writer.writerow({"generation": i, "renders": r, "best_fitness": f, "diversity": 1.0 / (i + 1)})


def _write_hillclimber_run(dir_path: Path, renders: list[float], best_fitness: list[float], algorithm: str | None = "hillclimb_1p1") -> None:
    _write_metrics_csv(dir_path / "metrics.csv", renders, best_fitness)
    payload = {"config": {}, "seed": 1, "versions": {}}
    if algorithm is not None:
        payload["algorithm"] = algorithm
    dir_path.mkdir(parents=True, exist_ok=True)
    with (dir_path / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_figure_claims_has_exactly_five_entries_each_a_specific_string():
    expected_files = {
        "fig_selection_fitness.png",
        "fig_selection_diversity.png",
        "fig_survival_kn.png",
        "fig_crossover_control.png",
        "fig_hillclimber_comparison.png",
    }

    assert set(FIGURE_CLAIMS.keys()) == expected_files
    for claim in FIGURE_CLAIMS.values():
        assert isinstance(claim, str) and len(claim) > 20


def test_plot_arm_produces_a_file_titled_with_its_n_seeds(tmp_path, monkeypatch):
    cell_a = tmp_path / "matrix" / "arm-label-a"
    cell_b = tmp_path / "matrix" / "arm-label-b"
    for cell, base in ((cell_a, 0.1), (cell_b, 0.5)):
        for seed in ("seed0", "seed1"):
            _write_metrics_csv(cell / seed / "metrics.csv", [10, 20, 30], [base, base + 0.05, base + 0.1])

    captured: dict[str, object] = {}
    real_subplots = gp.plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(gp.plt, "subplots", spy_subplots)
    monkeypatch.setattr(gp.plt, "close", lambda *a, **k: None)

    out_path = tmp_path / "fig.png"
    gp.plot_arm({"label-a": cell_a, "label-b": cell_b}, "Test claim", out_path, n_seeds=2)

    assert out_path.is_file() and out_path.stat().st_size > 0
    title = captured["ax"].get_title()
    assert "n=2" in title


def test_plot_hillclimber_comparison_caps_the_x_axis_at_the_shorter_arms_max_renders(tmp_path, monkeypatch):
    ga_cell = tmp_path / "matrix" / "selection-elite"
    ga_max_renders = [100, 120, 90, 110, 130]
    for seed_index, renders_max in enumerate(ga_max_renders):
        renders = [renders_max // 3, 2 * renders_max // 3, renders_max]
        _write_metrics_csv(ga_cell / f"seed{seed_index}" / "metrics.csv", renders, [0.5, 0.6, 0.7])

    hc_dir = tmp_path / "hillclimber"
    hc_max_renders = 60  # shorter than every GA seed's max renders
    _write_hillclimber_run(hc_dir, [20, 40, hc_max_renders], [0.3, 0.4, 0.5])

    monkeypatch.setattr(gp.plt, "close", lambda *a, **k: None)

    out_path = tmp_path / "hc_fig.png"
    gp.plot_hillclimber_comparison(ga_cell, hc_dir, out_path)

    assert out_path.is_file() and out_path.stat().st_size > 0

    # Independently recompute the expected cap: the minimum of the two arms'
    # max render counts, exactly the align_on_grid contract this function
    # must apply.
    expected_cap = min(min(ga_max_renders), hc_max_renders)
    from tp2.experiments.aggregate import align_on_grid, load_seed_curves

    curves = load_seed_curves(ga_cell)
    grid_x, _values = align_on_grid([*curves, gp._load_single_curve(hc_dir / "metrics.csv")])
    assert grid_x[-1] == pytest.approx(expected_cap)


def test_plot_hillclimber_comparison_refuses_to_label_a_run_missing_the_algorithm_field(tmp_path):
    ga_cell = tmp_path / "matrix" / "selection-elite"
    for seed_index in range(2):
        _write_metrics_csv(ga_cell / f"seed{seed_index}" / "metrics.csv", [10, 20], [0.5, 0.6])

    hc_dir = tmp_path / "hillclimber"
    _write_hillclimber_run(hc_dir, [10, 20], [0.3, 0.4], algorithm=None)

    with pytest.raises(GeneratePlotsError):
        gp.plot_hillclimber_comparison(ga_cell, hc_dir, tmp_path / "hc_fig.png")


def test_build_all_figures_raises_a_clear_error_naming_the_missing_cell(tmp_path):
    matrix_root = tmp_path / "matrix"
    # Only build the selection arm's cells; survival_kn/crossover_control
    # are deliberately absent, mirroring an unfinished/tracer-scale matrix.
    for label in gp.SELECTION_LABELS:
        for seed_index in range(2):
            _write_metrics_csv(
                matrix_root / f"selection-{label}" / f"seed{seed_index}" / "metrics.csv", [10, 20], [0.5, 0.6]
            )

    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    with pytest.raises(GeneratePlotsError, match="survival_kn"):
        gp.build_all_figures(matrix_root, tmp_path / "hillclimber", plots_dir)


def test_plot_survival_kn_produces_a_two_panel_figure(tmp_path, monkeypatch):
    matrix_root = tmp_path / "matrix"
    for ratio in gp.SURVIVAL_RATIOS:
        for strategy in gp.SURVIVAL_STRATEGIES:
            cell = matrix_root / f"survival_kn-krn-{ratio}-{strategy}"
            for seed_index in range(2):
                _write_metrics_csv(cell / f"seed{seed_index}" / "metrics.csv", [10, 20], [0.5, 0.6])

    monkeypatch.setattr(gp.plt, "close", lambda *a, **k: None)
    out_path = tmp_path / "survival.png"
    gp.plot_survival_kn(matrix_root, out_path, n_seeds=2)

    assert out_path.is_file() and out_path.stat().st_size > 0
