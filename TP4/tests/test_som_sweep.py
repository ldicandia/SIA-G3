"""Tracer tests: a SOM sweep runs its children and the som sweep_report hook selects one."""

from __future__ import annotations

import csv
import json

import pytest

from tp4.cli import main
from tp4.runs import PROJECT_ROOT

SOM_SWEEP = "configs/som_sweep.json"
PCA_SWEEP = "configs/pca_scaling_sweep.json"
TINY_GRID = {"params.rows": [3, 4], "params.epochs": [5, 10]}


def _rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def tiny_sweep(tmp_path_factory):
    """One shared tiny sweep (4 children, 2 workers) for the read-only assertions below."""
    root = tmp_path_factory.mktemp("som_sweep")
    cfg = root / "cfg.json"
    config = json.loads((PROJECT_ROOT / SOM_SWEEP).read_text(encoding="utf-8"))
    config["grid"] = TINY_GRID
    config["workers"] = 2
    cfg.write_text(json.dumps(config), encoding="utf-8")
    out = root / "runs"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    return out / "som_sweep"


def test_sweep_csv_has_one_row_per_child_with_the_four_metrics(tiny_sweep):
    rows = _rows(tiny_sweep / "sweep.csv")
    assert len(rows) == 4
    for column in ("quantization_error", "topographic_error", "distance_preservation", "empty_neuron_frac"):
        assert column in rows[0]
    assert [r["child"] for r in rows] == [
        "rows-3__epochs-5", "rows-3__epochs-10", "rows-4__epochs-5", "rows-4__epochs-10"
    ]


def test_every_child_has_its_own_run_dir_and_summary(tiny_sweep):
    for row in _rows(tiny_sweep / "sweep.csv"):
        assert (tiny_sweep / row["child"] / "summary.json").stat().st_size > 0


def test_selection_csv_marks_exactly_one_row(tiny_sweep):
    rows = _rows(tiny_sweep / "selection.csv")
    assert len(rows) == 4
    assert [r["selected"] for r in rows].count("True") == 1
    assert {"stage_occupancy", "stage_topology", "stage_order", "candidate_rank"} <= set(rows[0])


def test_selected_params_keep_the_grid_values_and_types(tiny_sweep):
    selected = json.loads((tiny_sweep / "selected_params.json").read_text(encoding="utf-8"))
    assert set(selected["params"]) == set(TINY_GRID)
    for key, value in selected["params"].items():
        assert any(v == value and type(v) is type(value) for v in TINY_GRID[key])


def test_summary_names_the_selected_child_and_its_values(tiny_sweep):
    summary = json.loads((tiny_sweep / "summary.json").read_text(encoding="utf-8"))
    chosen = next(r for r in _rows(tiny_sweep / "selection.csv") if r["selected"] == "True")
    assert summary["selected_child"] == chosen["child"]
    assert summary["selected_rows"] == int(chosen["params.rows"])
    assert summary["selected_epochs"] == int(chosen["params.epochs"])
    assert summary["rule_max_empty_frac"] == 0.5
    assert summary["rule_te_slack"] == 0.02
    assert summary["rule_dp_slack"] == 0.05
    assert {"n_children", "workers", "grid_keys"} <= set(summary)


def test_metrics_plot_is_written(tiny_sweep):
    assert (tiny_sweep / "sweep_metrics.png").stat().st_size > 0


def test_sweeps_of_kinds_without_a_report_hook_keep_the_phase_1_summary(tmp_path, write_config):
    cfg = write_config(PCA_SWEEP, child_plots=False, plots=False, workers=1)
    out = tmp_path / "runs"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    summary = json.loads((out / "pca_scaling_sweep" / "summary.json").read_text(encoding="utf-8"))
    assert summary == {"grid_keys": ["params.standardize"], "n_children": 2, "workers": 1}
