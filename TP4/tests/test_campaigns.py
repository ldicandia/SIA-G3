"""Experiment kind `campaigns`: evidence recomputed from the pca/som/oja configs, picks, baselines,
comparison, per-region strategy evidence and config validation."""

from __future__ import annotations

import copy
import csv
import json

import numpy as np
import pytest

from tp4.cli import main
from tp4.runs import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "datos" / "creadores.csv"
CAMPAIGNS = "configs/campaigns.json"
CLIENT_NAMES = ["dogs", "football", "makeup", "art"]
STRATEGIES = ["method", "followers_global", "followers_in_topic"]
EVIDENCE_COLUMNS = (
    "som_region", "som_row", "som_col", "oja_index", "oja_rank", "pc1", "pc2", "est_arg_reach",
    "cost_per_1000_arg_reach",
)
ARTIFACTS = (
    "run.json", "summary.json", "evidence.csv", "picks.csv", "baselines.csv", "comparison.csv",
    "strategy_profiles.csv", "comparison.png",
)


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary(run_dir):
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _config() -> dict:
    return json.loads((PROJECT_ROOT / CAMPAIGNS).read_text(encoding="utf-8"))


def _clients() -> list[dict]:
    return _config()["clients"]


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


def _expected_picks(evidence, client, n):
    """The rule recomputed from evidence.csv alone, with numpy: z over all rows (ddof=0), then the filter."""
    score = np.zeros(len(evidence))
    for item in client["rank_by"]:
        values = np.array([float(r[item["column"]]) for r in evidence])
        score += item["weight"] * (values - values.mean()) / values.std()
    regions = client.get("regions") or []
    keep = [
        float(r[client["topic"]]) >= client["min_share"] and (not regions or int(r["som_region"]) in regions)
        for r in evidence
    ]
    order = [i for i in np.argsort(-score, kind="stable") if keep[i]]
    return [evidence[i]["creator_id"] for i in order[:n]]


def _picked(run_dir, strategy_file="picks.csv"):
    by_client: dict[str, list[dict]] = {}
    for row in _read(run_dir / strategy_file):
        by_client.setdefault(row["client"], []).append(row)
    return by_client


# --------------------------------------------------------------------------- evidence


def test_cli_end_to_end_writes_every_artifact(run_dir):
    for name in ARTIFACTS:
        assert (run_dir / name).stat().st_size > 0, name


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


# --------------------------------------------------------------------------- picks


def test_picks_cover_the_four_clients_in_config_order(run_dir):
    picks = _read(run_dir / "picks.csv")
    assert len(picks) == 12
    assert [r["client"] for r in picks] == [n for n in CLIENT_NAMES for _ in range(3)]
    assert [c["name"] for c in _clients()] == CLIENT_NAMES
    for rows in _picked(run_dir).values():
        assert [int(r["pick_rank"]) for r in rows] == [1, 2, 3]
        scores = [float(r["score"]) for r in rows]
        assert scores == sorted(scores, reverse=True)
        for column in ("pc1", "pc2", "som_region", "som_row", "som_col", "oja_index", "oja_rank"):
            assert all(r[column] != "" for r in rows), column


def test_each_clients_picks_are_the_top_three_of_the_rule_recomputed_from_the_evidence(run_dir):
    evidence = _read(run_dir / "evidence.csv")
    by_client = _picked(run_dir)
    for client in _clients():
        rows = by_client[client["name"]]
        assert [r["creator_id"] for r in rows] == _expected_picks(evidence, client, 3), client["name"]
        assert all(float(r["topic_share"]) >= client["min_share"] for r in rows)
        if client.get("regions"):
            assert all(int(r["som_region"]) in client["regions"] for r in rows)


def test_dogs_are_filtered_to_region_1_and_the_others_have_no_region_filter():
    clients = {c["name"]: c for c in _clients()}
    assert clients["dogs"]["regions"] == [1]
    assert all(clients[n]["regions"] == [] for n in ("football", "makeup", "art"))
    assert clients["dogs"]["objective"] == "consultas y adopciones"
    assert all(clients[n]["objective"] == "ventas" for n in ("football", "makeup", "art"))


def test_every_pick_and_baseline_creator_exists_in_the_catalog(run_dir):
    known = {r["creator_id"] for r in _read(DATA_PATH)}
    for name in ("picks.csv", "baselines.csv"):
        assert {r["creator_id"] for r in _read(run_dir / name)} <= known


def test_also_picked_for_is_consistent_in_both_directions(run_dir):
    by_client = _picked(run_dir)
    picked_by = {}
    for client, rows in by_client.items():
        for row in rows:
            picked_by.setdefault(row["creator_id"], set()).add(client)
    for client, rows in by_client.items():
        for row in rows:
            listed = {c for c in row["also_picked_for"].split(";") if c}
            assert listed == picked_by[row["creator_id"]] - {client}, (client, row["name"])
    shared = {cid for cid, clients in picked_by.items() if len(clients) > 1}
    s = _summary(run_dir)
    assert s["n_overlapping_creators"] == len(shared)
    names = {r["creator_id"]: r["name"] for rows in by_client.values() for r in rows}
    assert sorted(s["overlapping_creators"]) == sorted(names[cid] for cid in shared)


# --------------------------------------------------------------------------- baselines


def test_followers_global_baseline_is_the_three_largest_accounts_for_every_client(run_dir):
    creators = _read(DATA_PATH)
    top = [r["creator_id"] for r in sorted(creators, key=lambda r: -float(r["followers"]))[:3]]  # stable ties
    rows = [r for r in _read(run_dir / "baselines.csv") if r["strategy"] == "followers_global"]
    assert len(rows) == 12
    for client in CLIENT_NAMES:
        mine = [r for r in rows if r["client"] == client]
        assert [r["creator_id"] for r in mine] == top
        assert [int(r["pick_rank"]) for r in mine] == [1, 2, 3]


def test_followers_in_topic_baseline_is_the_top_three_by_followers_among_the_topic_candidates(run_dir):
    evidence = _read(run_dir / "evidence.csv")
    rows = [r for r in _read(run_dir / "baselines.csv") if r["strategy"] == "followers_in_topic"]
    for client in _clients():
        candidates = [r for r in evidence if float(r[client["topic"]]) >= client["min_share"]]  # no region filter
        expected = [r["creator_id"] for r in sorted(candidates, key=lambda r: -float(r["followers"]))[:3]]
        assert [r["creator_id"] for r in rows if r["client"] == client["name"]] == expected, client["name"]


# --------------------------------------------------------------------------- comparison


def test_comparison_has_a_row_per_client_and_strategy_with_the_pooled_cost(run_dir):
    evidence = {r["name"]: r for r in _read(run_dir / "evidence.csv")}
    rows = _read(run_dir / "comparison.csv")
    assert len(rows) == 12
    assert [(r["client"], r["strategy"]) for r in rows] == [(c, s) for c in CLIENT_NAMES for s in STRATEGIES]
    clients = {c["name"]: c for c in _clients()}
    for row in rows:
        names = row["creators"].split(";")
        assert len(names) == 3
        picked = [evidence[n] for n in names]
        price = sum(float(r["price_usd"]) for r in picked)
        reach = sum(float(r["est_arg_reach"]) for r in picked)
        assert float(row["total_price_usd"]) == pytest.approx(price, rel=1e-6)
        assert float(row["total_est_arg_reach"]) == pytest.approx(reach, rel=1e-6)
        assert float(row["cost_per_1000_arg_reach"]) == pytest.approx(price / reach * 1000.0, rel=1e-6)
        topic = clients[row["client"]]["topic"]
        assert float(row["mean_topic_share"]) == pytest.approx(
            np.mean([float(r[topic]) for r in picked]), rel=1e-6
        )
        assert float(row["mean_argentina_audience_pct"]) == pytest.approx(
            np.mean([float(r["argentina_audience_pct"]) for r in picked]), rel=1e-6
        )
        assert float(row["mean_engagement_per_1000"]) == pytest.approx(
            np.mean([float(r["engagement_per_1000"]) for r in picked]), rel=1e-6
        )
    method = {r["client"]: set(r["creators"].split(";")) for r in rows if r["strategy"] == "method"}
    for row in rows:
        assert int(row["shared_with_method"]) == len(set(row["creators"].split(";")) & method[row["client"]])
        if row["strategy"] == "method":
            assert int(row["shared_with_method"]) == 3


def test_strategy_profiles_count_the_topic_candidates_by_region(run_dir):
    evidence = _read(run_dir / "evidence.csv")
    s = _summary(run_dir)
    profiles = _read(run_dir / "strategy_profiles.csv")
    for client in _clients():
        mine = [r for r in profiles if r["client"] == client["name"]]
        topic_candidates = [r for r in evidence if float(r[client["topic"]]) >= client["min_share"]]
        assert sum(int(r["n_candidates"]) for r in mine) == len(topic_candidates)
        in_regions = [int(r["n_candidates"]) for r in mine if not client["regions"] or int(r["som_region"]) in client["regions"]]
        assert sum(in_regions) == s[f"{client['name']}_n_candidates"]
        listed = [n for r in mine for n in r["creators"].split(";")]
        assert sorted(listed) == sorted(r["name"] for r in topic_candidates)
    football = {int(r["som_region"]) for r in profiles if r["client"] == "football"}
    assert football == {1, 2, 3}


def test_comparison_figure_is_written(run_dir):
    assert (run_dir / "comparison.png").stat().st_size > 0


def test_summary_carries_the_keys_section_1_4_cites(run_dir):
    s = _summary(run_dir)
    comparison = {(r["client"], r["strategy"]): r for r in _read(run_dir / "comparison.csv")}
    picks = _picked(run_dir)
    baselines = _picked(run_dir, "baselines.csv")
    for name in CLIENT_NAMES:
        for suffix in (
            "rule", "n_candidates", "picks", "baseline_topic", "mean_topic_share", "global_mean_topic_share",
            "topic_mean_topic_share", "cost_per_1000_arg_reach", "topic_cost_per_1000_arg_reach",
            "total_est_arg_reach", "mean_argentina_audience_pct", "mean_engagement_per_1000",
        ):
            assert f"{name}_{suffix}" in s, f"{name}_{suffix}"
        assert s[f"{name}_picks"] == [r["name"] for r in picks[name]]
        assert s[f"{name}_baseline_topic"] == [
            r["name"] for r in baselines[name] if r["strategy"] == "followers_in_topic"
        ]
        method, glob, topic = (comparison[(name, k)] for k in STRATEGIES)
        assert s[f"{name}_cost_per_1000_arg_reach"] == pytest.approx(float(method["cost_per_1000_arg_reach"]), abs=0.006)
        assert s[f"{name}_topic_cost_per_1000_arg_reach"] == pytest.approx(float(topic["cost_per_1000_arg_reach"]), abs=0.006)
        assert s[f"{name}_global_mean_topic_share"] == pytest.approx(float(glob["mean_topic_share"]), abs=0.006)
        assert s[f"{name}_topic_mean_topic_share"] == pytest.approx(float(topic["mean_topic_share"]), abs=0.006)
        assert s[f"{name}_mean_topic_share"] == pytest.approx(float(method["mean_topic_share"]), abs=0.006)
        assert s[f"{name}_total_est_arg_reach"] == round(float(method["total_est_arg_reach"]))
    glob = comparison[("dogs", "followers_global")]
    assert s["followers_global_picks"] == ["@mateconideas", "@golconmate", "@brillourbano"]
    assert s["followers_global_cost_per_1000_arg_reach"] == pytest.approx(float(glob["cost_per_1000_arg_reach"]), abs=0.006)
    assert s["followers_global_total_est_arg_reach"] == round(float(glob["total_est_arg_reach"]))
    assert s["followers_global_mean_argentina_audience_pct"] == pytest.approx(
        float(glob["mean_argentina_audience_pct"]), abs=0.006
    )
    assert s["n_creators"] == 50 and s["n_clients"] == 4 and s["n_picks"] == 3
    assert s["oja_config"] == "configs/oja.json" and s["n_regions"] == 3
    assert isinstance(s["oja_top_creator_picked_for"], list)
    top = s["oja_top_creator"]
    expected = [c for c in CLIENT_NAMES if top in s[f"{c}_picks"]]
    assert s["oja_top_creator_picked_for"] == expected
    assert s["dogs_rule"].startswith("topic_dogs_pct >= 40; región del mapa en [1]; puntaje = +1·z(")


# --------------------------------------------------------------------------- determinism


def test_the_same_config_twice_gives_byte_identical_tables(tmp_path, write_config):
    cfg = write_config(CAMPAIGNS, plots=False)
    for name in ("a", "b"):
        assert main(["run", str(cfg), "--out-root", str(tmp_path / name)]) == 0
    for table in ("evidence.csv", "picks.csv", "baselines.csv", "comparison.csv"):
        assert (tmp_path / "a" / "campaigns" / table).read_bytes() == (tmp_path / "b" / "campaigns" / table).read_bytes()


# --------------------------------------------------------------------------- validation


def _first(**changes) -> list[dict]:
    """The configured clients with `changes` applied to the first one."""
    clients = copy.deepcopy(_clients())
    clients[0].update(changes)
    return clients


def _duplicate_name() -> list[dict]:
    clients = copy.deepcopy(_clients())
    clients.append(copy.deepcopy(clients[0]))
    return clients


def _rank(column: str, weight) -> dict:
    return {"rank_by": [{"column": column, "weight": weight}]}


INVALID_CASES = [
    ("no_clients", {"clients": []}, "clients"),
    ("unknown_topic", {"clients": _first(topic="topic_cats_pct")}, "topic"),
    ("min_share_above_100", {"clients": _first(min_share=101)}, "min_share"),
    ("min_share_negative", {"clients": _first(min_share=-1)}, "min_share"),
    ("column_outside_allowlist", {"clients": _first(**_rank("price_ars", 1.0))}, "not rankable"),
    ("weight_zero", {"clients": _first(**_rank("saves_per_1000", 0))}, "weight"),
    ("empty_rank_by", {"clients": _first(rank_by=[])}, "rank_by"),
    ("duplicate_name", {"clients": _duplicate_name()}, "twice"),
    ("uppercase_name", {"clients": _first(name="Dogs")}, "name"),
    ("reserved_name", {"clients": _first(name="oja")}, "reserved"),
    ("unknown_client_key", {"clients": _first(budget=5)}, "unknown key"),
    ("region_zero", {"clients": _first(regions=[0])}, "regions"),
    ("n_picks_zero", {"params.n_picks": 0}, "n_picks"),
    ("n_picks_eleven", {"params.n_picks": 11}, "n_picks"),
    ("oja_config_of_another_kind", {"oja_config": "configs/som.json"}, "kind 'oja'"),
    ("region_above_the_map", {"clients": _first(regions=[9])}, "region"),  # run time: the map has 3 regions
    ("too_few_candidates", {"clients": _first(min_share=99, regions=[])}, "candidates"),  # run time
]


@pytest.mark.parametrize("case", INVALID_CASES, ids=[c[0] for c in INVALID_CASES])
def test_invalid_config_exits_2_and_leaves_no_run_dir(case, tmp_path, capsys, write_config):
    _, overrides, needle = case
    cfg = write_config(CAMPAIGNS, plots=False, **overrides)
    out = tmp_path / "bad_runs"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert needle in err
    assert not (out / "campaigns").exists()
