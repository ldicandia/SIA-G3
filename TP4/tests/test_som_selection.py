"""Selection rule edges, the trade-off plot, byte-identity across workers and the `check` command."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tp4.analysis.som_selection import (
    DP_SLACK,
    MAX_EMPTY_FRAC,
    TE_SLACK,
    rule_description,
    select_configuration,
)
from tp4.cli import main
from tp4.experiments.som_sweep_report import check_selection
from tp4.experiments.som_sweep_report import main as check_main
from tp4.runs import PROJECT_ROOT


def table(rows):
    """rows: dicts with qe/te/dp/empty and optional n/epochs; every other column is filled in."""
    return pd.DataFrame(
        [
            {
                "child": f"c{i}",
                "quantization_error": r.get("qe", 3.0),
                "topographic_error": r.get("te", 0.0),
                "distance_preservation": r.get("dp", 0.8),
                "empty_neuron_frac": r.get("empty", 0.2),
                "n_neurons": r.get("n", 36),
                "epochs": r.get("epochs", 200),
            }
            for i, r in enumerate(rows)
        ]
    )


def picked(rows):
    sel = select_configuration(table(rows))
    assert sel.table["selected"].sum() == 1
    return sel


def test_thresholds_are_the_documented_ones():
    assert (MAX_EMPTY_FRAC, TE_SLACK, DP_SLACK) == (0.5, 0.02, 0.05)
    text = rule_description()
    assert "0.5" in text and "0.02" in text and "0.05" in text


def test_lowest_qe_is_rejected_when_too_many_neurons_are_empty():
    rows = [{"qe": 2.0, "empty": 0.6}] + [{"qe": 3.0 - 0.1 * i, "empty": 0.1 * i} for i in range(7)]
    sel = picked(rows)
    assert sel.selected_position != 0
    assert not sel.table.loc[0, "stage_occupancy"]
    assert not sel.occupancy_fallback


def test_empty_fraction_exactly_half_passes_occupancy():
    sel = picked([{"qe": 2.0, "empty": 0.5}, {"qe": 3.0, "empty": 0.1}])
    assert sel.selected_position == 0
    assert sel.table.loc[0, "stage_occupancy"]


def test_te_edge_and_tolerance():
    sel = picked(
        [
            {"te": 0.0, "qe": 3.0},
            {"te": 0.02, "qe": 2.9},
            {"te": 0.04, "qe": 2.0},
            {"te": 0.020000000000000004, "qe": 2.8},
        ]
    )
    assert list(sel.table["stage_topology"]) == [True, True, False, True]
    assert sel.selected_position == 3


def test_dp_edge_nan_and_all_nan():
    sel = picked(
        [
            {"dp": 0.80, "qe": 3.0},
            {"dp": 0.75, "qe": 2.9},
            {"dp": 0.7499, "qe": 2.0},
            {"dp": np.nan, "qe": 1.0},
        ]
    )
    assert list(sel.table["stage_order"]) == [True, True, False, False]
    assert sel.selected_position == 1
    all_nan = picked([{"dp": np.nan, "qe": 3.0}, {"dp": np.nan, "qe": 2.0}])
    assert list(all_nan.table["stage_order"]) == [True, True]
    assert all_nan.selected_position == 1


def test_dp_none_cells_count_as_nan():
    df = table([{"qe": 3.0}, {"qe": 2.0}])
    df["distance_preservation"] = [0.8, None]
    sel = select_configuration(df)
    assert sel.selected_position == 0


def test_qe_ties_prefer_fewer_neurons_then_fewer_epochs_then_first_row():
    assert picked([{"qe": 2.0, "n": 64}, {"qe": 2.0, "n": 36}]).selected_position == 1
    assert picked([{"qe": 2.0, "epochs": 500}, {"qe": 2.0, "epochs": 50}]).selected_position == 1
    assert picked([{"qe": 2.0}, {"qe": 2.0}, {"qe": 2.0}]).selected_position == 0


def test_every_row_too_empty_falls_back_and_still_selects_one():
    sel = picked([{"empty": 0.7, "qe": 3.0}, {"empty": 0.7, "qe": 2.0}])
    assert sel.occupancy_fallback is True
    assert sel.selected_position == 1


def test_candidate_rank_follows_order_inside_stage_c_and_is_blank_elsewhere():
    sel = picked([{"qe": 2.5}, {"qe": 2.0}, {"qe": 1.0, "empty": 0.9}, {"qe": 2.2}])
    ranks = sel.table["candidate_rank"].tolist()
    assert ranks[1] == 1 and ranks[3] == 2 and ranks[0] == 3
    assert pd.isna(ranks[2])
    for column in ("stage_occupancy", "stage_topology", "stage_order", "selected"):
        assert sel.table[column].dtype == bool


# ---- check_selection / CLI ------------------------------------------------------------


def _sweep_dir(path, params, summary=None):
    path.mkdir(parents=True)
    (path / "selected_params.json").write_text(json.dumps({"child": "x", "params": params}), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps(summary or {}), encoding="utf-8")
    return path


def _config(path, params):
    path.write_text(json.dumps({"experiment": "som", "kind": "som", "seed": 1, "params": params}), encoding="utf-8")
    return path


def test_check_selection_consistent_returns_no_messages(tmp_path):
    cfg = _config(tmp_path / "som.json", {"rows": 8, "eta0": 0.5, "topology": "hex"})
    a = _sweep_dir(tmp_path / "a", {"params.topology": "hex"})
    b = _sweep_dir(tmp_path / "b", {"params.rows": 8, "params.eta0": 0.5})
    assert check_selection(cfg, [a, b]) == []


def test_check_selection_names_key_and_both_values_on_mismatch(tmp_path):
    cfg = _config(tmp_path / "som.json", {"rows": 6})
    sweep = _sweep_dir(tmp_path / "s", {"params.rows": 8})
    (message,) = check_selection(cfg, [sweep])
    assert message.startswith("MISMATCH")
    assert "params.rows" in message and "6" in message and "8" in message


def test_check_selection_reports_missing_selected_params(tmp_path):
    cfg = _config(tmp_path / "som.json", {"rows": 6})
    (message,) = check_selection(cfg, [tmp_path / "nope"])
    assert message.startswith("MISSING")


def test_check_selection_compares_run_metrics_with_last_sweep(tmp_path):
    cfg = _config(tmp_path / "som.json", {"rows": 6})
    sweep = _sweep_dir(
        tmp_path / "s",
        {"params.rows": 6},
        {"selected_quantization_error": 2.5, "selected_topographic_error": 0.0, "selected_distance_preservation": None},
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps({"quantization_error": 2.6, "topographic_error": 0.0, "distance_preservation": None}), encoding="utf-8"
    )
    messages = check_selection(cfg, [sweep], run)
    assert len(messages) == 1
    assert messages[0].startswith("METRIC quantization_error")
    (run / "summary.json").write_text(
        json.dumps({"quantization_error": 2.5, "topographic_error": 0.0, "distance_preservation": None}), encoding="utf-8"
    )
    assert check_selection(cfg, [sweep], run) == []


def test_check_main_exit_codes_and_output(tmp_path, capsys):
    cfg = _config(tmp_path / "som.json", {"rows": 6})
    sweep = _sweep_dir(tmp_path / "s", {"params.rows": 6})
    assert check_main(["check", str(cfg), str(sweep)]) == 0
    assert capsys.readouterr().out.startswith("ok:")
    bad = _sweep_dir(tmp_path / "bad", {"params.rows": 7})
    assert check_main(["check", str(cfg), str(bad)]) == 1
    assert "MISMATCH" in capsys.readouterr().out


# ---- report: trade-off plot and determinism -------------------------------------------


def _tiny_sweep(tmp_path, workers):
    config = json.loads((PROJECT_ROOT / "configs/som_sweep.json").read_text(encoding="utf-8"))
    config["grid"] = {"params.rows": [3, 4], "params.epochs": [5, 10]}
    config["workers"] = workers
    cfg = tmp_path / f"cfg{workers}.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / f"runs{workers}"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    return out / "som_sweep"


def test_tradeoff_plot_written_and_selection_identical_for_one_and_two_workers(tmp_path):
    one, two = _tiny_sweep(tmp_path, 1), _tiny_sweep(tmp_path, 2)
    assert (one / "sweep_tradeoff.png").stat().st_size > 0
    for name in ("selection.csv", "selected_params.json"):
        assert (one / name).read_bytes() == (two / name).read_bytes()


@pytest.mark.parametrize("plots", [False])
def test_no_pngs_when_plots_are_off(tmp_path, plots):
    config = json.loads((PROJECT_ROOT / "configs/som_sweep.json").read_text(encoding="utf-8"))
    config["grid"] = {"params.rows": [3, 4]}
    config["workers"] = 1
    config["plots"] = plots
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    assert main(["run", str(cfg), "--out-root", str(tmp_path / "runs")]) == 0
    sweep = tmp_path / "runs" / "som_sweep"
    assert not (sweep / "sweep_tradeoff.png").exists()
    assert not (sweep / "sweep_metrics.png").exists()
