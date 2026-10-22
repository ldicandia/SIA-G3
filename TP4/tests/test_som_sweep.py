"""SOM sweeps vary one parameter at a time and compare repeated trainings."""

from __future__ import annotations

import csv
import json
from pathlib import Path

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
    base = json.loads((PROJECT_ROOT / "configs/som.json").read_text())
    base["params"]["n_repeats"] = 2
    base_path = root / "base.json"
    base_path.write_text(json.dumps(base))
    config["base"] = str(base_path)
    config["grid"] = TINY_GRID
    config["workers"] = 2
    cfg.write_text(json.dumps(config), encoding="utf-8")
    out = root / "runs"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    return out / "som_sweep"


def test_sweep_csv_has_one_row_per_child_with_mean_and_sample_spread(tiny_sweep):
    rows = _rows(tiny_sweep / "sweep.csv")
    assert len(rows) == 4
    for metric in ("mean_bmu_distance", "u_mean", "u_max", "empty_neuron_frac"):
        assert f"{metric}_mean" in rows[0]
        assert f"{metric}_std" in rows[0]
    assert all(int(row["n_repeats"]) == 2 for row in rows)
    assert [r["child"] for r in rows] == [
        "rows-3", "rows-4", "epochs-5", "epochs-10"
    ]


def test_every_child_has_its_own_run_dir_and_summary(tiny_sweep):
    sweep = json.loads((tiny_sweep / "run.json").read_text())["config"]
    base = json.loads(Path(sweep["base"]).read_text())
    for row in _rows(tiny_sweep / "sweep.csv"):
        child_dir = tiny_sweep / row["child"]
        assert (child_dir / "summary.json").stat().st_size > 0
        assert len(_rows(child_dir / "per_seed.csv")) == 2
        child = json.loads((child_dir / "run.json").read_text())["config"]
        changed = {key for key in base["params"] if child["params"][key] != base["params"][key]}
        assert changed <= {row["varied_parameter"].removeprefix("params.")}
        assert float(row["parameter_value"]) == child["params"][row["varied_parameter"].removeprefix("params.")]
        assert len(changed) <= 1
    comparison = _rows(tiny_sweep / "comparison.csv")
    assert [row["varied_parameter"] for row in comparison] == ["params.rows"] * 2 + ["params.epochs"] * 2
    assert [int(row["params.rows"]) for row in comparison] == [3, 4, 4, 4]
    assert [int(row["params.epochs"]) for row in comparison] == [200, 200, 5, 10]


def test_metrics_plot_is_written(tiny_sweep):
    assert (tiny_sweep / "sweep_metrics.png").stat().st_size > 0


def test_sweeps_without_report_keep_the_base_summary(tmp_path, write_config):
    cfg = write_config(PCA_SWEEP, child_plots=False, plots=False, workers=1)
    out = tmp_path / "runs"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    summary = json.loads((out / "pca_scaling_sweep" / "summary.json").read_text(encoding="utf-8"))
    assert summary == {"grid_keys": ["params.standardize"], "n_children": 2, "workers": 1}
