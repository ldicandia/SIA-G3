"""Unit tests of the campaign selection rule, the followers baselines and the pooled metrics (toy frames)."""

from __future__ import annotations

import csv
import math

import numpy as np
import pandas as pd
import pytest

from tp4.analysis.campaigns import (
    DERIVED_COLUMNS,
    RANKABLE_COLUMNS,
    add_derived,
    catalog_zscores,
    describe_rule,
    followers_top,
    overlaps,
    pick,
    region_profiles,
    rule_scores,
    score_client,
    strategy_metrics,
)
from tp4.errors import ConfigError
from tp4.experiments.campaigns import _put
from tp4.runs import PROJECT_ROOT


def _toy(**columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    frame = pd.DataFrame(
        {"creator_id": [f"C{i}" for i in range(n)], "name": [f"@c{i}" for i in range(n)]}
        | {k: list(v) for k, v in columns.items()}
    )
    return frame


def _client(**changes) -> dict:
    client = {
        "name": "toy",
        "label": "Toy",
        "objective": "ventas",
        "topic": "share",
        "min_share": 40,
        "regions": [],
        "rank_by": [{"column": "x", "weight": 1.0}],
    }
    client.update(changes)
    return client


def test_catalog_zscores_use_ddof_zero():
    frame = _toy(x=[1.0, 2.0, 3.0, 4.0])
    z = catalog_zscores(frame, ["x"])["x"].to_numpy()
    assert z == pytest.approx((np.array([1, 2, 3, 4]) - 2.5) / math.sqrt(1.25))
    assert z.std(ddof=0) == pytest.approx(1.0)


def test_catalog_zscores_reject_a_constant_or_nan_column():
    with pytest.raises(ValueError, match="'flat'"):
        catalog_zscores(_toy(flat=[3.0, 3.0, 3.0]), ["flat"])
    with pytest.raises(ValueError, match="'gap'"):
        catalog_zscores(_toy(gap=[1.0, float("nan"), 3.0]), ["gap"])


def test_topic_filter_is_inclusive():
    frame = _toy(share=[40.0, 39.99, 80.0, 10.0], x=[1.0, 5.0, 3.0, 9.0], som_region=[1, 1, 1, 1])
    candidates = score_client(frame, _client())
    assert sorted(candidates["name"]) == ["@c0", "@c2"]
    assert sorted(candidates["topic_share"]) == [40.0, 80.0]


def test_a_region_list_filters_and_an_empty_or_absent_one_does_not():
    frame = _toy(share=[50.0] * 4, x=[1.0, 2.0, 3.0, 4.0], som_region=[1, 2, 3, 1])
    assert sorted(score_client(frame, _client(regions=[1]))["name"]) == ["@c0", "@c3"]
    assert sorted(score_client(frame, _client(regions=[1, 3]))["name"]) == ["@c0", "@c2", "@c3"]
    assert len(score_client(frame, _client(regions=[]))) == 4
    no_regions = _client()
    del no_regions["regions"]
    assert len(score_client(frame, no_regions)) == 4


def test_score_is_the_weighted_sum_of_catalog_zscores():
    frame = _toy(
        share=[50.0, 50.0, 50.0, 50.0], x=[1.0, 2.0, 3.0, 4.0], y=[4.0, 1.0, 3.0, 2.0], som_region=[1] * 4
    )
    client = _client(rank_by=[{"column": "x", "weight": 1.0}, {"column": "y", "weight": -0.5}])
    zx = (frame["x"] - frame["x"].mean()) / frame["x"].std(ddof=0)
    zy = (frame["y"] - frame["y"].mean()) / frame["y"].std(ddof=0)
    expected = (zx - 0.5 * zy).to_numpy()
    assert rule_scores(frame, client) == pytest.approx(expected)
    scored = score_client(frame, client)
    assert scored["score"].to_numpy() == pytest.approx(np.sort(expected)[::-1])


def test_scores_do_not_change_when_the_filter_drops_creators_but_the_catalog_keeps_them():
    frame = _toy(share=[45.0, 60.0, 90.0, 10.0, 70.0], x=[5.0, 1.0, 3.0, 8.0, 2.0], som_region=[1] * 5)
    wide = score_client(frame, _client(min_share=40)).set_index("name")["score"]
    narrow = score_client(frame, _client(min_share=65)).set_index("name")["score"]
    assert set(narrow.index) < set(wide.index)
    for name in narrow.index:
        assert narrow[name] == pytest.approx(wide[name])  # z comes from all five rows, not the candidates


def test_equal_scores_keep_catalog_order():
    frame = _toy(share=[50.0] * 5, x=[2.0, 7.0, 2.0, 7.0, 2.0], som_region=[1] * 5)
    assert score_client(frame, _client())["name"].tolist() == ["@c1", "@c3", "@c0", "@c2", "@c4"]
    assert pick(frame, _client(), 2)["name"].tolist() == ["@c1", "@c3"]


def test_followers_top_ties_keep_catalog_order_and_respect_the_mask():
    frame = _toy(followers=[100, 300, 300, 200, 300])
    top = followers_top(frame, 3)
    assert top["name"].tolist() == ["@c1", "@c2", "@c4"]
    assert top["pick_rank"].tolist() == [1, 2, 3]
    masked = followers_top(frame, 2, mask=np.array([True, False, False, True, True]))
    assert masked["name"].tolist() == ["@c4", "@c3"]
    with pytest.raises(ValueError):
        followers_top(frame, 6)


def test_pick_numbers_the_rows_and_needs_enough_candidates():
    frame = _toy(share=[50.0, 50.0, 10.0], x=[1.0, 2.0, 3.0], som_region=[1, 1, 1])
    picked = pick(frame, _client(), 2)
    assert picked["pick_rank"].tolist() == [1, 2]
    assert picked["name"].tolist() == ["@c1", "@c0"]
    with pytest.raises(ValueError, match="fewer than 3"):
        pick(frame, _client(), 3)


def test_strategy_metrics_pool_the_cost_and_average_the_shares():
    rows = _toy(
        share=[60.0, 80.0],
        argentina_audience_pct=[50.0, 90.0],
        est_arg_reach=[500.0, 3000.0],
        engagement_per_1000=[40.0, 60.0],
        price_usd=[10.0, 30.0],
        followers=[1000, 3000],
    )
    metrics = strategy_metrics(rows, "share")
    assert metrics["mean_topic_share"] == pytest.approx(70.0)
    assert metrics["mean_argentina_audience_pct"] == pytest.approx(70.0)
    assert metrics["total_est_arg_reach"] == pytest.approx(3500.0)
    assert metrics["mean_engagement_per_1000"] == pytest.approx(50.0)
    assert metrics["total_price_usd"] == pytest.approx(40.0)
    assert metrics["mean_followers"] == pytest.approx(2000.0)
    assert metrics["cost_per_1000_arg_reach"] == pytest.approx(40.0 / 3500.0 * 1000.0)  # pooled
    assert metrics["cost_per_1000_arg_reach"] != pytest.approx((20.0 + 10.0) / 2)  # not the mean of ratios


def test_overlaps_maps_a_creator_picked_by_two_clients_to_both():
    picks = {"dogs": ["A", "B", "C"], "art": ["C", "D", "E"], "makeup": ["C", "B", "F"]}
    assert overlaps(picks) == {"B": ["dogs", "makeup"], "C": ["art", "dogs", "makeup"]}
    assert overlaps({"a": ["X"], "b": ["Y"]}) == {}


def test_add_derived_reproduces_the_hand_computed_metrics():
    raw = pd.DataFrame(
        {
            "average_reach": [1000.0, 2000.0, 0.0],
            "argentina_audience_pct": [50.0, 25.0, 80.0],
            "likes_per_1000": [10.0, 20.0, 1.0],
            "comments_per_1000": [1.0, 2.0, 1.0],
            "saves_per_1000": [3.0, 4.0, 1.0],
            "shares_per_1000": [2.0, 3.0, 1.0],
            "price_usd": [100.0, 50.0, 70.0],
            "topic_football_pct": [10.0, 20.0, 30.0],
            "topic_makeup_pct": [20.0, 20.0, 30.0],
            "topic_dogs_pct": [30.0, 20.0, 30.0],
        }
    )
    derived = add_derived(raw)
    assert derived["est_arg_reach"].tolist() == [500.0, 500.0, 0.0]
    assert derived["engagement_per_1000"].tolist() == [16.0, 29.0, 4.0]
    assert derived["cost_per_1000_arg_reach"].iloc[:2].tolist() == pytest.approx([200.0, 100.0])
    assert math.isnan(derived["cost_per_1000_arg_reach"].iloc[2])  # no Argentine reach: no cost per reach
    assert derived["topic_art_pct"].tolist() == [40.0, 40.0, 10.0]
    assert "est_arg_reach" not in raw.columns  # the input is not modified


def test_describe_rule_uses_decimal_commas():
    client = _client(
        topic="topic_dogs_pct",
        regions=[1],
        rank_by=[{"column": "argentina_audience_pct", "weight": 1.0}, {"column": "saves_per_1000", "weight": -0.5}],
    )
    assert describe_rule(client) == (
        "topic_dogs_pct >= 40; región del mapa en [1]; puntaje = +1·z(argentina_audience_pct) -0,5·z(saves_per_1000)"
    )
    assert "." not in describe_rule(_client(min_share=42.5, regions=[]))
    assert "región" not in describe_rule(_client(regions=[]))


def test_region_profiles_group_the_topic_candidates_by_region():
    frame = _toy(
        share=[50.0, 60.0, 70.0, 10.0, 90.0],
        som_region=[1, 1, 3, 2, 3],
        followers=[100, 300, 500, 700, 900],
        est_arg_reach=[1000.0, 3000.0, 2000.0, 9999.0, 2000.0],
        argentina_audience_pct=[60.0, 80.0, 40.0, 10.0, 20.0],
        non_follower_reach_pct=[10.0, 30.0, 50.0, 0.0, 70.0],
        engagement_per_1000=[10.0, 30.0, 50.0, 0.0, 70.0],
        saves_per_1000=[1.0, 3.0, 5.0, 0.0, 7.0],
        profile_visits_per_1000=[2.0, 4.0, 6.0, 0.0, 8.0],
        content_tutorial_pct=[20.0, 40.0, 10.0, 0.0, 30.0],
        price_usd=[10.0, 30.0, 20.0, 99.0, 20.0],
        oja_index=[1.0, 3.0, -1.0, 0.0, -3.0],
    )
    profile = region_profiles(frame, _client(regions=[1]))  # the region filter is NOT applied
    assert profile["som_region"].tolist() == [1, 3]  # region 2 has no topic candidate: omitted
    assert profile["n_candidates"].tolist() == [2, 2]
    assert profile["creators"].tolist() == ["@c0;@c1", "@c2;@c4"]
    assert profile["client"].tolist() == ["toy", "toy"]
    first = profile.iloc[0]
    assert first["mean_followers"] == pytest.approx(200.0)
    assert first["mean_est_arg_reach"] == pytest.approx(2000.0)
    assert first["mean_content_tutorial_pct"] == pytest.approx(30.0)
    assert first["cost_per_1000_arg_reach"] == pytest.approx(40.0 / 4000.0 * 1000.0)
    assert first["mean_oja_index"] == pytest.approx(2.0)


def test_put_refuses_a_duplicate_summary_key():
    summary: dict = {}
    _put(summary, "n_picks", 3)
    with pytest.raises(ConfigError, match="n_picks"):
        _put(summary, "n_picks", 4)


def test_rankable_columns_are_unique_and_exist():
    assert len(set(RANKABLE_COLUMNS)) == len(RANKABLE_COLUMNS)
    assert "price_ars" not in RANKABLE_COLUMNS and "organic_posts_pct" not in RANKABLE_COLUMNS
    with open(PROJECT_ROOT / "datos" / "creadores.csv", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    known = set(header) | set(DERIVED_COLUMNS) | {"pc1", "pc2", "oja_index"}
    assert set(RANKABLE_COLUMNS) <= known
    assert set(DERIVED_COLUMNS) <= set(RANKABLE_COLUMNS)
