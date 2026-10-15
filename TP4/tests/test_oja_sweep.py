"""Oja eta sweep: the PC1-free selection rule, the sweep artifacts and the link to configs/oja.json."""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
import pytest

from tp4.cli import main
from tp4.experiments.oja_sweep_report import (
    RANK_DISPLACEMENT_MAX,
    max_rank_displacement,
    rule_description,
    select_eta,
)
from tp4.runs import PROJECT_ROOT

SELECTION_COLUMNS = [
    "eta0", "n_seeds", "n_diverged", "max_epochs_to_converge", "max_norm_deviation",
    "max_rank_displacement", "min_cosine_to_pc1", "stage_stable", "stage_norm", "stage_index", "selected",
]


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- the rule on hand-built tables


def _ranks(order):
    """Series creator_id -> rank for creators listed best first."""
    return pd.Series(range(1, len(order) + 1), index=list(order))


SAME = _ranks(["a", "b", "c", "d"])
SWAPPED = _ranks(["a", "c", "b", "d"])


def _row(eta, seed, *, diverged=False, epochs=10, norm=1.002, cos=0.9999, flipped=False):
    return {
        "child": f"eta0-{eta}__seed-{seed}",
        "params.eta0": eta,
        "seed": seed,
        "diverged": diverged,
        "epochs_to_converge": np.nan if (diverged or epochs is None) else float(epochs),
        "final_w_norm": np.nan if diverged else norm,
        "cosine_to_pc1": np.nan if diverged else cos,
        "orientation_flipped": None if diverged else flipped,
    }


def _table_and_rankings(spec):
    """spec: {eta: {seed: row kwargs, ... ,'ranks': {seed: Series}}}"""
    rows, rankings = [], {}
    for eta, seeds in spec.items():
        for seed, kwargs in seeds.items():
            ranks = kwargs.pop("ranks", SAME)
            row = _row(eta, seed, **kwargs)
            rows.append(row)
            if not kwargs.get("diverged"):
                rankings[row["child"]] = ranks
    return pd.DataFrame(rows), rankings


def test_each_stage_rejects_its_own_failure_and_the_fastest_survivor_is_selected():
    table, rankings = _table_and_rankings(
        {
            1e-5: {1: dict(epochs=None), 2: dict(epochs=None)},  # never inside the band
            2e-4: {1: dict(), 2: dict(diverged=True)},  # a diverged seed
            5e-4: {1: dict(epochs=9), 2: dict(epochs=12, ranks=SWAPPED)},  # rankings differ
            1e-3: {1: dict(epochs=40), 2: dict(epochs=44)},  # passes everything, slow
            2e-3: {1: dict(epochs=5), 2: dict(epochs=7)},  # passes everything, fastest
        }
    )
    sel = select_eta(table, rankings)
    t = sel.table
    assert list(t.columns) == SELECTION_COLUMNS
    assert t["eta0"].tolist() == [1e-5, 2e-4, 5e-4, 1e-3, 2e-3]
    assert t["stage_stable"].tolist() == [True, False, True, True, True]
    assert t["stage_norm"].tolist() == [False, False, True, True, True]
    assert t["stage_index"].tolist() == [True, True, False, True, True]
    assert t["max_rank_displacement"].tolist()[2] == 1
    assert sel.selected_eta0 == 2e-3
    assert t["selected"].tolist() == [False, False, False, False, True]


def test_a_tie_on_epochs_goes_to_the_smaller_eta():
    table, rankings = _table_and_rankings(
        {
            1e-3: {1: dict(epochs=6), 2: dict(epochs=8)},
            5e-4: {1: dict(epochs=8), 2: dict(epochs=3)},
        }
    )
    assert select_eta(table, rankings).selected_eta0 == 5e-4


def test_nothing_selected_when_no_eta_passes_and_nothing_raises():
    table, rankings = _table_and_rankings(
        {0.5: {1: dict(diverged=True), 2: dict(diverged=True)}, 0.02: {1: dict(epochs=None), 2: dict(epochs=None)}}
    )
    sel = select_eta(table, rankings)
    assert sel.selected_eta0 is None
    assert not sel.table["selected"].any()
    assert sel.table["n_diverged"].tolist() == [2, 0]


def test_max_rank_displacement():
    assert RANK_DISPLACEMENT_MAX == 0
    assert max_rank_displacement([SAME, SAME.copy(), SAME.copy()]) == 0
    assert max_rank_displacement([SAME, SWAPPED]) == 1
    assert max_rank_displacement([SAME]) == 0
    far = _ranks(["c", "a", "b", "d"])
    assert max_rank_displacement([SAME, far]) == 2


def test_the_rule_never_reads_pc1_and_describes_itself():
    # same table, different PC1 columns: the selection must not change (prohibition OJA-02)
    table, rankings = _table_and_rankings({1e-3: {1: dict(epochs=6), 2: dict(epochs=6)}, 2e-3: {1: dict(epochs=7), 2: dict(epochs=7)}})
    other = table.copy()
    other["cosine_to_pc1"] = 0.1
    assert select_eta(table, rankings).selected_eta0 == select_eta(other, rankings).selected_eta0 == 1e-3
    text = rule_description()
    assert isinstance(text, str) and len(text) > 40


# --------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    base = tmp_path_factory.mktemp("oja_sweep")
    out = base / "runs"
    assert main(["run", "configs/oja_sweep.json", "--out-root", str(out)]) == 0
    direct = json.loads((PROJECT_ROOT / "configs" / "oja.json").read_text(encoding="utf-8"))
    direct["plots"] = False
    direct_path = base / "oja_direct.json"
    direct_path.write_text(json.dumps(direct), encoding="utf-8")
    assert main(["run", str(direct_path), "--out-root", str(out)]) == 0
    return out


def test_sweep_writes_every_artifact_and_27_children(sweep):
    run_dir = sweep / "oja_sweep"
    for name in ("sweep.csv", "selection.csv", "selected_params.json", "sweep_convergence.png", "sign_ambiguity.png", "summary.json"):
        assert (run_dir / name).stat().st_size > 0, name
    assert len(_read(run_dir / "sweep.csv")) == 27
    assert len([p for p in run_dir.glob("eta0-*") if p.is_dir()]) == 27


def test_selection_table_one_row_per_eta_in_grid_order(sweep):
    grid = _json(PROJECT_ROOT / "configs" / "oja_sweep.json")["grid"]["params.eta0"]
    rows = _read(sweep / "oja_sweep" / "selection.csv")
    assert list(rows[0]) == SELECTION_COLUMNS
    assert [float(r["eta0"]) for r in rows] == grid
    by_eta = {float(r["eta0"]): r for r in rows}
    assert by_eta[0.06]["stage_stable"] == "False" and by_eta[0.5]["stage_stable"] == "False"
    selected = [r for r in rows if r["selected"] == "True"]
    assert len(selected) == 1
    assert selected[0]["stage_stable"] == selected[0]["stage_norm"] == selected[0]["stage_index"] == "True"


def test_the_four_regimes_show_up_in_the_sweep(sweep):
    rows = {float(r["eta0"]): r for r in _read(sweep / "oja_sweep" / "selection.csv")}
    assert rows[1e-5]["stage_norm"] == "False"  # too slow for 200 epochs
    assert rows[5e-4]["stage_norm"] == "True"  # converges
    assert rows[0.005]["stage_norm"] == "False" and rows[0.005]["stage_stable"] == "True"  # oscillates outside the band
    assert int(rows[0.06]["n_diverged"]) == 3 and int(rows[0.5]["n_diverged"]) == 3  # diverges


def test_the_final_config_carries_the_selected_eta(sweep):
    summary = _json(sweep / "oja_sweep" / "summary.json")
    config = _json(PROJECT_ROOT / "configs" / "oja.json")
    assert summary["selected_eta0"] is not None
    assert config["params"]["eta0"] == summary["selected_eta0"]
    selected = _json(sweep / "oja_sweep" / "selected_params.json")
    assert selected["params"] == {"params.eta0": summary["selected_eta0"]}
    assert selected["child"] == summary["selected_child"] and summary["selected_child"].endswith("__seed-42")
    assert isinstance(summary["selection_rule"], str) and summary["selection_rule"]
    assert summary["n_eta"] == 9 and summary["seeds"] == [42, 7, 11]


def test_the_selected_child_reproduces_a_direct_run(sweep):
    summary = _json(sweep / "oja_sweep" / "summary.json")
    child = _json(sweep / "oja_sweep" / summary["selected_child"] / "summary.json")
    direct = _json(sweep / "oja" / "summary.json")
    for key in ("cosine_to_pc1", "pearson_scores_pc1", "final_w_norm", "epochs_to_converge", "top_creator"):
        assert child[key] == direct[key], key
    assert (sweep / "oja_sweep" / summary["selected_child"] / "ranking.csv").read_bytes() == (sweep / "oja" / "ranking.csv").read_bytes()


def test_sign_ambiguity_at_the_selected_eta(sweep):
    summary = _json(sweep / "oja_sweep" / "summary.json")
    assert summary["seeds_flipped"] and summary["seeds_not_flipped"]
    assert sorted(summary["seeds_flipped"] + summary["seeds_not_flipped"]) == sorted(summary["seeds"])
    # ... yet every seed gives the same creator order after orientation
    eta_dirs = [p for p in (sweep / "oja_sweep").iterdir() if p.is_dir() and p.name.startswith(summary["selected_child"].split("__")[0])]
    assert len(eta_dirs) == 3
    orders = [[r["creator_id"] for r in _read(p / "ranking.csv")] for p in eta_dirs]
    assert orders[0] == orders[1] == orders[2]


def test_oja_sweep_report_is_registered_on_the_kind():
    from tp4.experiments import get_experiment
    from tp4.experiments.oja_sweep_report import report

    assert get_experiment("oja").sweep_report is report
