"""Experiment kind `oja`: end-to-end CLI run, validation evidence, ranking and config validation."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from tp4 import runner
from tp4.cli import main
from tp4.config import load_config
from tp4.experiments.oja import MAX_EPOCHS, fit_index, rank_order
from tp4.models.oja import OjaConfig
from tp4.runs import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "datos" / "creadores.csv"
ARTIFACTS = (
    "run.json", "summary.json", "ranking.csv", "history.csv", "weights_vs_pc1.csv",
    "convergence.png", "cos_to_pc1.png", "weights_vs_pc1.png", "scores_vs_pc1.png",
    "repeats.csv", "history_repeats.csv", "history_stats.csv", "weights_repeats.csv",
    "weights_stats.csv", "index_repeats.csv", "index_stats.csv",
)
RANKING_HEADER = [
    "rank", "creator_id", "name", "dominant_topic", "topic_football_pct", "topic_makeup_pct",
    "topic_dogs_pct", "topic_art_pct", "followers", "oja_index", "pc1_score",
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
    base = tmp_path_factory.mktemp("oja_runs")
    out = base / "runs"
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


def test_ranking_retains_raw_art_share_and_topics_sum_to_100(oja_run):
    by_id = {r["creator_id"]: r for r in _read(DATA_PATH)}
    for row in _read(oja_run / "ranking.csv"):
        raw = by_id[row["creator_id"]]
        assert float(row["followers"]) == float(raw["followers"])
        total = sum(float(row[c]) for c in ("topic_football_pct", "topic_makeup_pct", "topic_dogs_pct", "topic_art_pct"))
        assert total == pytest.approx(100.0, abs=0.025)
        assert float(row["topic_art_pct"]) == float(raw["topic_art_pct"])


def test_weights_vs_pc1_table(oja_run):
    rows = _read(oja_run / "weights_vs_pc1.csv")
    assert list(rows[0]) == ["feature", "oja_weight", "pc1_loading", "difference", "oja_weight_raw"]
    assert len(rows) == 27
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
    assert isinstance(s["norm_band_epoch"], int) and s["norm_band_epoch"] <= s["epochs"]
    assert abs(s["final_w_norm"] - 1.0) <= p["norm_tol"]
    assert s["lambda1"] > 0 and s["course_eta_bound"] > p["eta0"]
    assert s["max_abs_weight_diff"] < 0.01
    assert s["initial_w_norm"] > 1.0 and s["final_delta_w"] >= 0.0
    assert len(s["top5_creators"]) == len(s["bottom5_creators"]) == 5
    assert s["top_creator"] == s["top5_creators"][0] and s["bottom_creator"] == s["bottom5_creators"][-1]
    assert s["init"] == "uniform01" and s["input"] == "features" and s["n_features"] == 27


def test_history_has_one_row_per_epoch(oja_run):
    rows = _read(oja_run / "history.csv")
    assert list(rows[0]) == ["epoch", "eta", "w_norm", "delta_w", "cos_to_pc1"]
    assert len(rows) == _config()["params"]["epochs"]
    assert [int(r["epoch"]) for r in rows] == list(range(1, len(rows) + 1))


def test_rank_order_ties_keep_csv_order():
    assert rank_order([1.0, 2.0, 2.0, 0.5]).tolist() == [1, 2, 0, 3]


def test_same_seed_gives_byte_identical_tables(tmp_path, write_config):
    cfg = write_config(base="configs/oja.json", plots=False, **{"params.n_repeats": 2})
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert main(["run", str(cfg), "--out-root", str(out_a)]) == 0
    assert main(["run", str(cfg), "--out-root", str(out_b)]) == 0
    for name in ("ranking.csv", "history.csv", "weights_vs_pc1.csv", "summary.json",
                 "repeats.csv", "history_stats.csv", "weights_stats.csv", "index_stats.csv"):
        assert (out_a / "oja" / name).read_bytes() == (out_b / "oja" / name).read_bytes(), name


def test_diverging_eta_is_flagged_and_writes_no_ranking(tmp_path, write_config):
    cfg = write_config(base="configs/oja.json", **{"params.eta0": 0.5})
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    s = _summary(out / "oja")
    assert s["diverged"] is True and isinstance(s["divergence_epoch"], int)
    assert s["cosine_to_pc1"] is None and s["pearson_scores_pc1"] is None
    assert s["norm_band_epoch"] is None and s["final_w_norm"] is None
    assert not (out / "oja" / "ranking.csv").exists()
    assert (out / "oja" / "history.csv").stat().st_size > 0
    assert (out / "oja" / "convergence.png").stat().st_size > 0


INVALID_CASES = [
    {"params.epochs": 0},
    {"params.epochs": MAX_EPOCHS + 1},
    {"params.eta0": 0},
    {"params.eta0": 1.5},
    {"params.eta_min": 0},
    {"params.eta_min": 0.01},  # above the default eta0 0.001
    {"params.eta_decay": "cosine"},
    {"params.n_repeats": 1},
    {"params.n_repeats": 201},
    {"params.n_repeats": True},
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


@pytest.mark.parametrize(
    "overrides",
    [
        {"params.eta0": 1.0},
        {"params.epochs": MAX_EPOCHS},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_boundary_values_pass_validation(write_config, overrides):
    cfg = write_config(base="configs/oja.json", **overrides)
    config = runner.validate_config(load_config(cfg))
    assert config["kind"] == "oja"


def test_training_repetitions_use_distinct_seeds_and_preserve_the_reference_run(oja_run):
    rows = _read(oja_run / "repeats.csv")
    assert len(rows) == 30
    assert [int(row["seed"]) for row in rows] == list(range(42, 72))
    assert all(row["diverged"] == "False" for row in rows)
    values = np.array([float(row["cosine_to_pc1"]) for row in rows])
    summary = _summary(oja_run)
    assert len(set(values)) > 1
    assert float(rows[0]["final_w_norm"]) == pytest.approx(summary["final_w_norm"], abs=0.00005)
    assert summary["cosine_to_pc1_mean"] == pytest.approx(values.mean())
    assert summary["cosine_to_pc1_std"] == pytest.approx(values.std(ddof=1))
    assert summary["cosine_to_pc1_n"] == summary["n_successful_repeats"] == 30
    assert summary["n_diverged_repeats"] == 0


def test_weight_and_index_error_bars_match_seed_dispersion(oja_run):
    weights = _read(oja_run / "weights_repeats.csv")
    weight_stats = _read(oja_run / "weights_stats.csv")
    for row in weight_stats:
        values = np.array([float(w["oja_weight"]) for w in weights if w["feature"] == row["feature"]])
        assert len(values) == int(row["n_repeats"]) == 30
        assert float(row["oja_weight_mean"]) == pytest.approx(values.mean(), abs=1e-9)
        assert float(row["oja_weight_std"]) == pytest.approx(values.std(ddof=1), abs=1e-9)
    indices = _read(oja_run / "index_repeats.csv")
    stats = _read(oja_run / "index_stats.csv")
    for row in stats:
        creator = [r for r in indices if r["creator_id"] == row["creator_id"]]
        values = np.array([float(r["oja_index"]) for r in creator])
        assert float(row["oja_index_mean"]) == pytest.approx(values.mean(), abs=1e-8)
        assert float(row["oja_index_std"]) == pytest.approx(values.std(ddof=1), abs=1e-8)
        assert int(row["top5_count"]) == sum(int(r["rank"]) <= 5 for r in creator)
    assert sum(int(row["top5_count"]) for row in stats) == 30 * 5


def test_training_curve_reports_mean_sample_sd_and_count_per_epoch(oja_run):
    histories = _read(oja_run / "history_repeats.csv")
    stats = _read(oja_run / "history_stats.csv")
    assert len(stats) == _config()["params"]["epochs"]
    for row in (stats[0], stats[-1]):
        epoch = int(float(row["epoch"]))
        values = np.array([float(h["w_norm"]) for h in histories if int(float(h["epoch"])) == epoch])
        assert int(row["w_norm_n"]) == len(values) == 30
        assert float(row["w_norm_mean"]) == pytest.approx(values.mean(), abs=1e-9)
        assert float(row["w_norm_std"]) == pytest.approx(values.std(ddof=1), abs=1e-9)


def test_reference_orients_reported_index_without_changing_training():
    X = np.array([[1.0, 0.3], [-1.0, -0.3], [0.5, -0.2], [-0.5, 0.2]])
    config = OjaConfig(epochs=10, eta0=0.001)
    original = fit_index(X, config, np.random.default_rng(8), reference=None)
    aligned = fit_index(X, config, np.random.default_rng(8), reference=-original.w_hat)
    np.testing.assert_array_equal(original.result.weights, aligned.result.weights)
    np.testing.assert_allclose(aligned.w_hat, -original.w_hat)
    np.testing.assert_allclose(aligned.index, -original.index)
