"""Experiment kind `oja`: end-to-end CLI run, validation evidence, ranking and config validation."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from tp4 import runner
from tp4.cli import main
from tp4.config import load_config
from tp4.experiments.oja import MAX_EPOCHS, rank_order
from tp4.runs import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "datos" / "creadores.csv"
ARTIFACTS = (
    "run.json", "summary.json", "ranking.csv", "history.csv", "weights_vs_pc1.csv",
    "convergence.png", "cos_to_pc1.png", "weights_vs_pc1.png", "scores_vs_pc1.png",
)
RANKING_HEADER = [
    "rank", "creator_id", "name", "dominant_topic", "topic_football_pct", "topic_makeup_pct",
    "topic_dogs_pct", "topic_art_pct", "followers", "oja_index", "pc1_score", "som_region",
]


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary(run_dir):
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _config() -> dict:
    return json.loads((PROJECT_ROOT / "configs" / "oja.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """configs/som.json (no plots) and configs/oja.json run into one out root."""
    base = tmp_path_factory.mktemp("oja_runs")
    out = base / "runs"
    som = json.loads((PROJECT_ROOT / "configs" / "som.json").read_text(encoding="utf-8"))
    som["plots"] = False
    som_path = base / "som_noplots.json"
    som_path.write_text(json.dumps(som), encoding="utf-8")
    assert main(["run", str(som_path), "--out-root", str(out)]) == 0
    assert main(["run", "configs/oja.json", "--out-root", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def oja_run(runs):
    return runs / "oja"


def test_cli_end_to_end_writes_every_artifact(oja_run):
    for name in ARTIFACTS:
        assert (oja_run / name).stat().st_size > 0, name


def test_ranking_header_rows_and_order(oja_run):
    creators = _read(DATA_PATH)
    ranking = _read(oja_run / "ranking.csv")
    assert list(ranking[0]) == RANKING_HEADER
    assert len(ranking) == len(creators)
    assert [int(r["rank"]) for r in ranking] == list(range(1, len(creators) + 1))
    index = [float(r["oja_index"]) for r in ranking]
    assert index == sorted(index, reverse=True)
    assert sorted(r["creator_id"] for r in ranking) == sorted(r["creator_id"] for r in creators)
    assert ranking[0]["name"] == "@mundoanimal.ar"


def test_ranking_art_share_is_reconstructed_and_topics_sum_to_100(oja_run):
    by_id = {r["creator_id"]: r for r in _read(DATA_PATH)}
    for row in _read(oja_run / "ranking.csv"):
        raw = by_id[row["creator_id"]]
        assert float(row["followers"]) == float(raw["followers"])
        total = sum(float(row[c]) for c in ("topic_football_pct", "topic_makeup_pct", "topic_dogs_pct", "topic_art_pct"))
        assert total == pytest.approx(100.0, abs=1e-6)


def test_som_region_equals_the_som_run_regions(runs):
    som_regions = {r["creator_id"]: int(r["region"]) for r in _read(runs / "som" / "bmus.csv")}
    ranking = _read(runs / "oja" / "ranking.csv")
    assert {r["creator_id"]: int(r["som_region"]) for r in ranking} == som_regions


def test_weights_vs_pc1_table(oja_run):
    rows = _read(oja_run / "weights_vs_pc1.csv")
    assert list(rows[0]) == ["feature", "oja_weight", "pc1_loading", "difference", "oja_weight_raw"]
    assert len(rows) == 25
    w = np.array([float(r["oja_weight"]) for r in rows])
    pc1 = np.array([float(r["pc1_loading"]) for r in rows])
    diff = np.array([float(r["difference"]) for r in rows])
    assert float((w**2).sum()) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(diff, w - pc1, atol=1e-8)
    raw = np.array([float(r["oja_weight_raw"]) for r in rows])
    assert abs(np.linalg.norm(raw) - 1.0) <= 0.01  # the trained norm, before normalization


def test_summary_validation_numbers_and_orientation(oja_run):
    s = _summary(oja_run)
    p = _config()["params"]
    assert s["diverged"] is False and s["divergence_epoch"] is None
    assert s["cosine_to_pc1"] >= 0.999 and s["pearson_scores_pc1"] >= 0.999
    assert s["cosine_to_pc1"] <= 1.0
    assert s["orientation_matches_pc1"] is True
    assert isinstance(s["orientation_flipped"], bool)
    assert s["oja_top_feature"] == s["pc1_top_feature"] == "average_caption_words"
    assert isinstance(s["epochs_to_converge"], int) and s["epochs_to_converge"] <= s["epochs"]
    assert abs(s["final_w_norm"] - 1.0) <= p["norm_tol"]
    assert s["lambda1"] == 13.2944 and s["course_eta_bound"] == 0.0627
    assert s["max_abs_weight_diff"] < 0.01
    assert s["initial_w_norm"] > 1.0 and s["final_delta_w"] >= 0.0
    assert len(s["top5_creators"]) == len(s["bottom5_creators"]) == 5
    assert s["top_creator"] == s["top5_creators"][0] and s["bottom_creator"] == s["bottom5_creators"][-1]
    assert s["som_config"] == "configs/som.json" and s["n_regions"] >= 2
    assert s["init"] == "uniform01" and s["input"] == "features" and s["n_features"] == 25


def test_history_has_one_row_per_epoch(oja_run):
    rows = _read(oja_run / "history.csv")
    assert list(rows[0]) == ["epoch", "eta", "w_norm", "delta_w", "cos_to_pc1"]
    assert len(rows) == _config()["params"]["epochs"]
    assert [int(r["epoch"]) for r in rows] == list(range(1, len(rows) + 1))


def test_rank_order_ties_keep_csv_order():
    assert rank_order([1.0, 2.0, 2.0, 0.5]).tolist() == [1, 2, 0, 3]


def test_same_seed_gives_byte_identical_tables(tmp_path, write_config):
    cfg = write_config(base="configs/oja.json", plots=False)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert main(["run", str(cfg), "--out-root", str(out_a)]) == 0
    assert main(["run", str(cfg), "--out-root", str(out_b)]) == 0
    for name in ("ranking.csv", "history.csv", "weights_vs_pc1.csv", "summary.json"):
        assert (out_a / "oja" / name).read_bytes() == (out_b / "oja" / name).read_bytes(), name


# --------------------------------------------------------------------------- divergence


def test_diverging_eta_is_flagged_and_writes_no_ranking(tmp_path, write_config):
    cfg = write_config(base="configs/oja.json", **{"params.eta0": 0.5})
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    s = _summary(out / "oja")
    assert s["diverged"] is True and isinstance(s["divergence_epoch"], int)
    assert s["cosine_to_pc1"] is None and s["pearson_scores_pc1"] is None
    assert s["epochs_to_converge"] is None and s["final_w_norm"] is None
    assert not (out / "oja" / "ranking.csv").exists()
    assert (out / "oja" / "history.csv").stat().st_size > 0
    assert (out / "oja" / "convergence.png").stat().st_size > 0


# --------------------------------------------------------------------------- validation

INVALID_CASES = [
    {"params.epochs": 0},
    {"params.epochs": MAX_EPOCHS + 1},
    {"params.eta0": 0},
    {"params.eta0": 1.5},
    {"params.eta_min": 0},
    {"params.eta_min": 0.01},  # above the default eta0 0.001
    {"params.eta_decay": "cosine"},
    {"params.norm_tol": 0},
    {"params.norm_tol": 1},
    {"params.unknown": 1},
    {"params.epochs": True},
]


@pytest.mark.parametrize("overrides", INVALID_CASES, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()))
def test_invalid_oja_config_exits_2_and_creates_no_run_dir(tmp_path, write_config, capsys, overrides):
    cfg = write_config(base="configs/oja.json", **overrides)
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not (out / "oja").exists()


def test_som_config_of_another_kind_is_rejected(tmp_path, write_config, capsys):
    cfg = write_config(base="configs/oja.json", som_config="configs/pca.json")
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    err = capsys.readouterr().err
    assert "error:" in err and "som" in err
    assert not (out / "oja").exists()


def test_som_config_with_a_different_feature_set_is_rejected(tmp_path, write_config, capsys):
    other = tmp_path / "feature_set_copy.json"
    other.write_text((PROJECT_ROOT / "configs" / "feature_set.json").read_text(encoding="utf-8"), encoding="utf-8")
    som = json.loads((PROJECT_ROOT / "configs" / "som.json").read_text(encoding="utf-8"))
    som["feature_set"] = str(other)
    som_path = tmp_path / "som_other.json"
    som_path.write_text(json.dumps(som), encoding="utf-8")
    cfg = write_config(base="configs/oja.json", som_config=str(som_path))
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    err = capsys.readouterr().err
    assert "error:" in err and "feature_set" in err
    assert not (out / "oja").exists()


def test_som_config_with_a_different_data_file_is_rejected(tmp_path, write_config, capsys):
    other = tmp_path / "creadores_copy.csv"
    other.write_text(DATA_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    som = json.loads((PROJECT_ROOT / "configs" / "som.json").read_text(encoding="utf-8"))
    som["data"] = str(other)
    som_path = tmp_path / "som_other_data.json"
    som_path.write_text(json.dumps(som), encoding="utf-8")
    cfg = write_config(base="configs/oja.json", som_config=str(som_path))
    assert main(["run", str(cfg), "--out-root", str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert "error:" in err and "data" in err


def test_som_config_must_be_a_non_empty_string(tmp_path, write_config, capsys):
    cfg = write_config(base="configs/oja.json", som_config="")
    assert main(["run", str(cfg), "--out-root", str(tmp_path / "out")]) == 2
    assert "som_config" in capsys.readouterr().err


@pytest.mark.parametrize(
    "overrides",
    [
        {"params.eta0": 1.0},
        {"params.eta_min": 0.001, "params.eta0": 0.001},
        {"params.epochs": MAX_EPOCHS},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_boundary_values_pass_validation(write_config, overrides):
    cfg = write_config(base="configs/oja.json", **overrides)
    config = runner.validate_config(load_config(cfg))
    assert config["kind"] == "oja"
