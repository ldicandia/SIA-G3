"""Experiment kind `campaigns`: evidence recomputed from the pca/som/oja configs, picks and baselines."""

from __future__ import annotations

import csv
import json

import pytest

from tp4.cli import main
from tp4.runs import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "datos" / "creadores.csv"
CAMPAIGNS = "configs/campaigns.json"
EVIDENCE_COLUMNS = (
    "som_region", "som_row", "som_col", "oja_index", "oja_rank", "pc1", "pc2", "est_arg_reach",
    "cost_per_1000_arg_reach",
)


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary(run_dir):
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _config() -> dict:
    return json.loads((PROJECT_ROOT / CAMPAIGNS).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """som and oja (no plots) and campaigns run into one out root, so the evidence can be compared."""
    base = tmp_path_factory.mktemp("campaign_runs")
    out = base / "runs"
    for name in ("som", "oja"):
        config = json.loads((PROJECT_ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))
        config["plots"] = False
        path = base / f"{name}_noplots.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        assert main(["run", str(path), "--out-root", str(out)]) == 0
    assert main(["run", CAMPAIGNS, "--out-root", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def run_dir(runs):
    return runs / "campaigns"


def test_evidence_has_one_row_per_creator_in_csv_order(run_dir):
    creators = _read(DATA_PATH)
    evidence = _read(run_dir / "evidence.csv")
    assert [r["creator_id"] for r in evidence] == [r["creator_id"] for r in creators]
    assert [r["name"] for r in evidence] == [r["name"] for r in creators]
    assert all(column in evidence[0] for column in EVIDENCE_COLUMNS)
    assert len({r["creator_id"] for r in evidence}) == len(creators)


def test_evidence_equals_the_oja_and_som_runs(runs):
    evidence = {r["creator_id"]: r for r in _read(runs / "campaigns" / "evidence.csv")}
    ranking = {r["creator_id"]: r for r in _read(runs / "oja" / "ranking.csv")}
    bmus = {r["creator_id"]: r for r in _read(runs / "som" / "bmus.csv")}
    assert set(evidence) == set(ranking) == set(bmus)
    for creator_id, row in evidence.items():
        assert float(row["oja_index"]) == float(ranking[creator_id]["oja_index"])
        assert float(row["pc1"]) == float(ranking[creator_id]["pc1_score"])
        assert int(row["som_region"]) == int(bmus[creator_id]["region"])
        assert int(row["som_region"]) == int(ranking[creator_id]["som_region"])


def test_oja_rank_matches_the_oja_ranking_order(runs):
    ranking = _read(runs / "oja" / "ranking.csv")
    rank_of = {r["creator_id"]: int(r["rank"]) for r in ranking}
    for row in _read(runs / "campaigns" / "evidence.csv"):
        assert int(row["oja_rank"]) == rank_of[row["creator_id"]]


def test_dogs_picks_follow_the_rule_and_carry_the_evidence(run_dir):
    picks = _read(run_dir / "picks.csv")
    assert [r["client"] for r in picks] == ["dogs"] * 3
    assert [int(r["pick_rank"]) for r in picks] == [1, 2, 3]
    assert all(float(r["topic_share"]) >= 40 for r in picks)
    assert all(int(r["som_region"]) == 1 for r in picks)
    scores = [float(r["score"]) for r in picks]
    assert scores == sorted(scores, reverse=True)
    for column in ("pc1", "pc2", "som_row", "som_col", "oja_index", "oja_rank"):
        assert all(r[column] != "" for r in picks), column


def test_every_pick_and_baseline_creator_exists_in_the_catalog(run_dir):
    known = {r["creator_id"] for r in _read(DATA_PATH)}
    for name in ("picks.csv", "baselines.csv"):
        assert {r["creator_id"] for r in _read(run_dir / name)} <= known


def test_followers_global_baseline_is_the_three_largest_accounts(run_dir):
    creators = _read(DATA_PATH)
    top = sorted(creators, key=lambda r: -float(r["followers"]))[:3]  # stable: ties keep CSV order
    baseline = [r for r in _read(run_dir / "baselines.csv") if r["strategy"] == "followers_global"]
    assert [r["creator_id"] for r in baseline] == [r["creator_id"] for r in top]
    assert [int(r["pick_rank"]) for r in baseline] == [1, 2, 3]


def test_summary_names_match_the_csv_rows(run_dir):
    s = _summary(run_dir)
    assert s["dogs_picks"] == [r["name"] for r in _read(run_dir / "picks.csv") if r["client"] == "dogs"]
    assert s["followers_global_picks"] == [
        r["name"] for r in _read(run_dir / "baselines.csv") if r["strategy"] == "followers_global"
    ]
    assert s["followers_global_picks"] == ["@mateconideas", "@golconmate", "@brillourbano"]
    assert s["n_creators"] == 50 and s["n_picks"] == 3 and s["oja_config"] == "configs/oja.json"
    assert s["dogs_rule"].startswith("topic_dogs_pct >= 40; región del mapa en [1]; puntaje = +1·z(")
