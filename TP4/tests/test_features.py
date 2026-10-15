from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tp4.cli import main
from tp4.data import load_creators
from tp4.errors import ConfigError
from tp4.features import (
    IDENTITIES,
    FeatureSet,
    check_identities,
    load_design,
    load_feature_set,
    near_redundant_pairs,
    select_features,
    selected_columns,
    validate_feature_set,
)

FEATURE_SET = "configs/feature_set.json"
DATA = "datos/creadores.csv"
EXPECTED_DROPS = {
    "price_ars",
    "total_engagement_per_1000",
    "estimated_argentina_reach",
    "organic_posts_pct",
    "topic_art_pct",
    "content_showcase_pct",
}


@pytest.fixture(scope="module")
def creators() -> pd.DataFrame:
    return load_creators(DATA)


@pytest.fixture(scope="module")
def fs() -> FeatureSet:
    return load_feature_set(FEATURE_SET)


def _write_fs(tmp_path, payload) -> str:
    path = tmp_path / "fs.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_declared_feature_set_has_six_reasoned_drops(fs):
    assert set(fs.drop) == EXPECTED_DROPS
    assert all(isinstance(r, str) and r.strip() for r in fs.drop.values())
    assert set(fs.reconstruct) == {"topic_art_pct", "content_showcase_pct"}
    assert fs.source == FEATURE_SET


def test_identities_are_the_six_documented_ones():
    assert [i.name for i in IDENTITIES] == [
        "price_ars_ratio",
        "total_engagement_sum",
        "argentina_reach_product",
        "sponsored_organic_complement",
        "topic_block",
        "content_block",
    ]


def test_check_identities_on_real_data_all_hold_and_are_broken_by_the_feature_set(creators, fs):
    report = check_identities(creators, fs)
    assert len(report) == 6
    assert report["holds"].all()
    assert (report["broken_by"] != "").all()
    by_name = report.set_index("name")["broken_by"]
    assert by_name["price_ars_ratio"] == "price_ars"
    assert by_name["topic_block"] == "topic_art_pct"
    assert by_name["content_block"] == "content_showcase_pct"
    assert list(report.columns) == [
        "name", "kind", "columns", "formula", "max_abs_residual", "tolerance", "holds", "broken_by", "member_std",
    ]


def test_check_identities_without_feature_set_has_empty_broken_by(creators):
    assert (check_identities(creators)["broken_by"] == "").all()


def _reach_frame(estimated: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"average_reach": [100.0], "argentina_audience_pct": [50.0], "estimated_argentina_reach": [estimated]}
    )


def test_residual_exactly_at_tolerance_holds():
    report = check_identities(_reach_frame(51.0)).set_index("name")
    row = report.loc["argentina_reach_product"]
    assert row["max_abs_residual"] == 1.0 and row["tolerance"] == 1.0
    assert bool(row["holds"]) is True
    assert bool(check_identities(_reach_frame(51.5)).set_index("name").loc["argentina_reach_product", "holds"]) is False


def test_identity_with_missing_columns_does_not_hold():
    row = check_identities(_reach_frame(50.0)).set_index("name").loc["price_ars_ratio"]
    assert bool(row["holds"]) is False
    assert row["formula"].endswith(" (columns missing)")


def test_near_redundant_pairs_threshold_is_inclusive_and_sorted():
    cols = ["a", "b", "c"]
    corr = pd.DataFrame(
        [[1.0, 0.95, -0.95], [0.95, 1.0, 0.5], [-0.95, 0.5, 1.0]], index=cols, columns=cols
    )
    pairs = near_redundant_pairs(corr, 0.95, ["a", "b"])
    assert list(pairs.columns) == ["column_a", "column_b", "r", "abs_r", "both_selected"]
    assert list(zip(pairs["column_a"], pairs["column_b"])) == [("a", "b"), ("a", "c")]
    assert list(pairs["both_selected"]) == [True, False]
    assert list(pairs["r"]) == [0.95, -0.95]
    assert near_redundant_pairs(corr, 0.9500001, ["a"]).empty


def test_near_redundant_pairs_sorted_by_descending_abs_r():
    cols = ["x", "y", "z"]
    corr = pd.DataFrame(
        [[1.0, 0.96, 0.99], [0.96, 1.0, -0.97], [0.99, -0.97, 1.0]], index=cols, columns=cols
    )
    pairs = near_redundant_pairs(corr, 0.9, cols)
    assert list(zip(pairs["column_a"], pairs["column_b"])) == [("x", "z"), ("y", "z"), ("x", "y")]


def test_near_redundant_real_data_leads_with_reach_video_views(creators):
    from tp4.analysis.eda import correlation_matrix
    from tp4.data import numeric_columns

    cols = numeric_columns(creators)
    pairs = near_redundant_pairs(
        correlation_matrix(creators, cols), 0.95, [c for c in cols if c not in EXPECTED_DROPS]
    )
    # The full matrix leads with the exact-identity pairs (|r| = 1, a dropped member), not selected.
    assert pairs.iloc[0]["abs_r"] == pytest.approx(1.0) and not pairs.iloc[0]["both_selected"]
    first = pairs[pairs["both_selected"]].iloc[0]
    assert {first["column_a"], first["column_b"]} == {"average_reach", "average_video_views"}
    assert first["abs_r"] == pytest.approx(0.997, abs=1e-3)


def test_validate_feature_set_names_the_unbroken_identity(creators):
    drop = {c: "x" for c in EXPECTED_DROPS - {"topic_art_pct"}}
    with pytest.raises(ConfigError, match="topic_block"):
        validate_feature_set(creators, FeatureSet(drop=drop, source="t"))
    with pytest.raises(ConfigError, match="price_ars_ratio"):
        validate_feature_set(creators, FeatureSet(drop={}, source="t"))


def test_select_features_refuses_invalid_sets_and_accepts_the_declared_one(creators, fs):
    bad = FeatureSet(drop={"price_ars": "x"}, source="t")
    with pytest.raises(ConfigError, match="topic_block"):
        select_features(creators, bad)
    X = select_features(creators, fs)
    assert X.shape == (len(creators), 25)
    assert list(X.columns) == selected_columns(creators, fs)
    numeric = [c for c in creators.columns if c not in ("creator_id", "name")]
    assert list(X.columns) == [c for c in numeric if c not in EXPECTED_DROPS]
    assert "topic_art_pct" not in X.columns and "topic_football_pct" in X.columns


@pytest.mark.parametrize("bad_col", ["not_a_column", "name", "creator_id"])
def test_selected_columns_rejects_unknown_or_id_columns(creators, bad_col):
    with pytest.raises(ConfigError, match=bad_col):
        selected_columns(creators, FeatureSet(drop={bad_col: "x"}, source="t"))


def test_selected_columns_rejects_non_numeric_column():
    df = pd.DataFrame({"creator_id": [1, 2], "name": ["a", "b"], "txt": ["u", "v"], "n": [1.0, 2.0]})
    # load_creators coerces everything to numeric, so a text column can only reach here from a hand-built frame.
    with pytest.raises(ConfigError, match="txt"):
        selected_columns(df.astype({"txt": "object"}), FeatureSet(drop={"txt": "x"}, source="t"))


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"drop": {"price_ars": ""}}, "reason"),
        ({"drop": {"price_ars": "   "}}, "reason"),
        ({"drop": {"price_ars": 3}}, "reason"),
        ({"drop": {"price_ars": "ok"}, "extra": 1}, "unknown key"),
        ({"drop": {}}, "non-empty"),
        ({"reconstruct": {}}, "non-empty"),
        ({"drop": {"price_ars": "ok"}, "reconstruct": {"topic_art_pct": "f"}}, "not in 'drop'"),
        ([], "object"),
    ],
)
def test_load_feature_set_rejects_malformed_files(tmp_path, payload, match):
    with pytest.raises(ConfigError, match=match):
        load_feature_set(_write_fs(tmp_path, payload))


def test_load_feature_set_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_feature_set(str(path))


def test_load_design_keeps_every_creator_and_ids(creators):
    design = load_design(DATA, FEATURE_SET)
    assert len(design.X) == len(design.raw) == len(creators)
    assert list(design.ids.columns) == ["creator_id", "name"]
    assert design.X.index.equals(design.raw.index)


def test_pca_run_with_an_invalid_feature_set_fails_before_runs_is_created(tmp_path, write_config, capsys):
    bad = _write_fs(tmp_path, {"drop": {"price_ars": "x"}})
    cfg = write_config("configs/pca.json", plots=False, feature_set=bad)
    assert main(["run", str(cfg), "--out-root", str(tmp_path / "runs")]) == 2
    assert "topic_block" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "pca").exists()


def test_eda_cli_proves_the_feature_set(tmp_path):
    out = tmp_path / "runs"
    assert main(["run", "configs/eda.json", "--out-root", str(out)]) == 0
    run = out / "eda"
    report = pd.read_csv(run / "identities.csv")
    assert len(report) == 6 and report["holds"].all() and report["broken_by"].notna().all()
    resolved = json.loads((run / "feature_set.json").read_text(encoding="utf-8"))
    assert resolved["valid"] is True and resolved["n_selected"] == 25
    assert "topic_art_pct" not in resolved["selected"] and "topic_dogs_pct" in resolved["selected"]
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected_rank"] == summary["n_selected_features"] == 25
    assert summary["numeric_rank"] == 26
    assert summary["n_identities"] == 6 and summary["n_identities_holding"] == 6
    pairs = pd.read_csv(run / "near_redundant.csv")
    kept = pairs[pairs["both_selected"]].iloc[0]
    assert {kept["column_a"], kept["column_b"]} == {"average_reach", "average_video_views"}
    assert (np.diff(pairs["abs_r"].to_numpy()) <= 1e-12).all()
    assert (run / "correlation_heatmap_selected.png").stat().st_size > 0


def test_eda_reports_but_does_not_enforce_an_invalid_feature_set(tmp_path, write_config):
    only_price = _write_fs(tmp_path, {"drop": {"price_ars": "x"}})
    cfg = write_config(feature_set=only_price, plots=False)
    out = tmp_path / "runs"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    resolved = json.loads((out / "eda" / "feature_set.json").read_text(encoding="utf-8"))
    assert resolved["valid"] is False
    report = pd.read_csv(out / "eda" / "identities.csv").set_index("name")
    assert report.loc["price_ars_ratio", "broken_by"] == "price_ars"
    assert pd.isna(report.loc["topic_block", "broken_by"])


def test_eda_with_a_malformed_feature_set_still_fails(tmp_path, write_config, capsys):
    bad = _write_fs(tmp_path, {"drop": {"price_ars": ""}})
    cfg = write_config(feature_set=bad, plots=False)
    assert main(["run", str(cfg), "--out-root", str(tmp_path / "runs")]) == 2
    assert "error:" in capsys.readouterr().err
