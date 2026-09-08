"""Unit tests for scripts/generate_plots.py's seven-figure set.

All fixtures are hand-written, tiny, synthetic metrics.csv/run.json files
under tmp_path -- never a dependency on any real (gitignored) run directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
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


def _write_full_metrics_csv(
    path: Path,
    generations: list[int],
    best_fitness: list[float],
    elapsed_s: list[float],
    worst_fitness: list[float],
) -> None:
    """A metrics.csv carrying the columns the bar/cost figures read.

    `_write_metrics_csv` above deliberately stays minimal (the curve figures
    only need generation/renders/best_fitness/diversity); the bar and cost
    figures additionally read `elapsed_s` and `worst_fitness`, so they get
    their own fixture writer rather than widening every existing test's
    fixture and changing what those tests exercise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["generation", "renders", "best_fitness", "worst_fitness", "diversity", "elapsed_s"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for gen, best, elapsed, worst in zip(generations, best_fitness, elapsed_s, worst_fitness):
            writer.writerow({
                "generation": gen,
                "renders": gen * 10,
                "best_fitness": best,
                "worst_fitness": worst,
                "diversity": 1.0 / (gen + 1),
                "elapsed_s": elapsed,
            })


def _write_hillclimber_run(dir_path: Path, renders: list[float], best_fitness: list[float], algorithm: str | None = "hillclimb_1p1") -> None:
    _write_metrics_csv(dir_path / "metrics.csv", renders, best_fitness)
    payload = {"config": {}, "seed": 1, "versions": {}}
    if algorithm is not None:
        payload["algorithm"] = algorithm
    dir_path.mkdir(parents=True, exist_ok=True)
    with (dir_path / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_figure_claims_has_exactly_the_expected_entries_each_a_specific_string():
    expected_files = {
        "fig_selection_fitness.png",
        "fig_selection_diversity.png",
        "fig_selection_parents_fitness.png",
        "fig_selection_parents_diversity.png",
        "fig_survival_kn.png",
        "fig_crossover_control.png",
        "fig_hillclimber_comparison.png",
        "fig_mutation_fitness.png",
        "fig_mutation_final_bars.png",
        "fig_selection_final_bars.png",
        "fig_cost.png",
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
    # Only build both selection arms' cells; survival_kn/crossover_control
    # are deliberately absent, mirroring an unfinished/tracer-scale matrix.
    # `selection_parents-*` must be present too: build_all_figures checks it
    # BEFORE survival_kn, so omitting it would make this test fail on the
    # wrong missing cell and stop exercising the survival_kn failure it
    # was written to prove.
    for label in gp.SELECTION_LABELS:
        for arm in ("selection", "selection_parents"):
            for seed_index in range(2):
                _write_metrics_csv(
                    matrix_root / f"{arm}-{label}" / f"seed{seed_index}" / "metrics.csv", [10, 20], [0.5, 0.6]
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


def test_mutation_labels_match_the_shipped_mutation_matrix_spec():
    """MUTATION_LABELS must name exactly the cells the spec actually ships.

    Read from the spec file rather than restated, the same way SELECTION_LABELS
    is kept honest: a label added to the JSON and forgotten here would silently
    drop a cell from the figure.
    """
    spec_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "mutation_matrix.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert set(gp.MUTATION_LABELS) == set(spec["arms"]["mutation"])
    assert set(gp.MUTATION_EXPECTED_GENES) == set(gp.MUTATION_LABELS)


def test_generations_to_threshold_reports_nan_when_a_seed_never_reaches_it(tmp_path):
    cell = tmp_path / "mutation-complete"
    # seed0 crosses 0.95 at generation 2; seed1 plateaus below it forever.
    _write_full_metrics_csv(
        cell / "seed0" / "metrics.csv", [0, 1, 2], [0.70, 0.90, 0.96], [1.0, 2.0, 3.0], [0.60, 0.61, 0.62]
    )
    _write_full_metrics_csv(
        cell / "seed1" / "metrics.csv", [0, 1, 2], [0.70, 0.71, 0.72], [1.0, 2.0, 3.0], [0.60, 0.61, 0.62]
    )

    reached = gp._generations_to_threshold(cell, 0.95)

    assert reached[0] == pytest.approx(2.0)
    # The plateauing seed must NOT be silently recorded as "reached it on the
    # last generation" -- that is the failure mode this column exists to avoid.
    assert np.isnan(reached[1])


def _write_run_json(path: Path, image: str, canvas: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"config": {"image": image, "canvas": canvas}, "seed": 0, "versions": {}}, handle)


def test_blank_canvas_fitness_matches_the_engine_on_an_all_zero_genome(tmp_path):
    """The floor must be what the ENGINE scores for a blank frame.

    Recomputed here independently through Evaluator rather than compared to a
    hardcoded 0.695, so the test still means something if the fitness
    definition or the shipped target ever changes.
    """
    import numpy as np

    from tp2.engine.fitness import Evaluator
    from tp2.engine.genome import chromosome_length
    from tp2.io.images import load_target

    project_root = Path(__file__).resolve().parents[1]
    image_rel = "assets/flag_ar.png"
    canvas = 64
    matrix_root = tmp_path / "matrix"
    _write_run_json(matrix_root / "selection-elite" / "seed0" / "run.json", image_rel, canvas)

    monkey_root = gp.PROJECT_ROOT
    assert (monkey_root / image_rel).is_file(), "the shipped target must exist for this test"

    size = (canvas, canvas)
    expected, _frame = Evaluator(load_target(monkey_root / image_rel, size), size).evaluate(
        np.zeros(chromosome_length(1), dtype=np.float32)
    )

    assert gp.blank_canvas_fitness(matrix_root) == pytest.approx(float(expected))


def test_blank_canvas_fitness_is_independent_of_the_triangle_budget(tmp_path):
    """Budget 1 is not a shortcut that changes the answer.

    An all-zero genome draws nothing at any budget, which is the reason
    blank_canvas_fitness may hardcode chromosome_length(1) instead of
    recovering a budget run.json never archives.
    """
    import numpy as np

    from tp2.engine.fitness import Evaluator
    from tp2.engine.genome import chromosome_length
    from tp2.io.images import load_target

    size = (64, 64)
    target = load_target(gp.PROJECT_ROOT / "assets" / "flag_ar.png", size)
    scores = {
        budget: float(Evaluator(target, size).evaluate(np.zeros(chromosome_length(budget), dtype=np.float32))[0])
        for budget in (1, 7, 30)
    }
    assert len(set(scores.values())) == 1, scores


def test_blank_canvas_fitness_errors_when_no_cell_archived_a_run_json(tmp_path):
    matrix_root = tmp_path / "matrix"
    _write_metrics_csv(matrix_root / "selection-elite" / "seed0" / "metrics.csv", [10, 20], [0.5, 0.6])

    with pytest.raises(GeneratePlotsError, match="run.json"):
        gp.blank_canvas_fitness(matrix_root)


def test_plot_final_bars_baselines_the_x_axis_at_the_given_floor(tmp_path, monkeypatch):
    cells = {}
    for label, final in (("elite", 0.98), ("roulette", 0.72)):
        cell = tmp_path / f"selection-{label}"
        for seed in range(2):
            _write_full_metrics_csv(
                cell / f"seed{seed}" / "metrics.csv",
                [0, 1], [0.70, final], [1.0, 2.0], [0.69, 0.70],
            )
        cells[label] = cell

    captured: dict[str, object] = {}
    real_subplots = gp.plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(gp.plt, "subplots", spy_subplots)
    monkeypatch.setattr(gp.plt, "close", lambda *a, **k: None)

    out_path = tmp_path / "bars.png"
    gp.plot_final_bars(cells, "Test claim", out_path, baseline=0.695, baseline_label="piso", n_seeds=2)

    assert out_path.is_file() and out_path.stat().st_size > 0
    ax = captured["ax"]
    # Baselined at the floor, never at zero and never at an arbitrary crop.
    assert ax.get_xlim()[0] == pytest.approx(0.695)
    # Sorted ascending, so the best method is the top bar.
    assert [label.get_text() for label in ax.get_yticklabels()] == ["roulette", "elite"]


def test_plot_cost_labels_a_cell_that_never_reaches_the_threshold(tmp_path, monkeypatch):
    cells = {}
    for label, best in (("elite", [0.70, 0.96]), ("roulette", [0.70, 0.72])):
        cell = tmp_path / f"selection-{label}"
        for seed in range(2):
            _write_full_metrics_csv(
                cell / f"seed{seed}" / "metrics.csv", [0, 1], best, [1.0, 40.0], [0.69, 0.70]
            )
        cells[label] = cell

    monkeypatch.setattr(gp.plt, "close", lambda *a, **k: None)
    out_path = tmp_path / "cost.png"
    gp.plot_cost(cells, "Test claim", out_path, threshold=0.95, n_seeds=2)

    assert out_path.is_file() and out_path.stat().st_size > 0
    # The plateauing cell must be named on the figure, not dropped: assert on
    # the underlying computation the annotation is driven by.
    assert np.isnan(gp._generations_to_threshold(cells["roulette"], 0.95)).all()


def test_build_all_figures_skips_the_mutation_pair_when_no_mutation_root_is_given(tmp_path):
    """mutation_root=None must skip, not fail.

    Proven by the error raised being about the FIRST missing main-matrix cell,
    never about a missing mutation cell -- i.e. the mutation branch was not
    entered at all.
    """
    matrix_root = tmp_path / "matrix"
    for label in gp.SELECTION_LABELS:
        for arm in ("selection", "selection_parents"):
            for seed_index in range(2):
                _write_metrics_csv(
                    matrix_root / f"{arm}-{label}" / f"seed{seed_index}" / "metrics.csv", [10, 20], [0.5, 0.6]
                )

    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    with pytest.raises(GeneratePlotsError) as excinfo:
        gp.build_all_figures(matrix_root, tmp_path / "hillclimber", plots_dir, mutation_root=None)
    assert "mutation_matrix.json" not in str(excinfo.value)
