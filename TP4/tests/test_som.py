"""Experiment kind `som`: end-to-end CLI run, artifacts and config validation."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from tp4.cli import main
from tp4.experiments.som import labels_by_neuron
from tp4.runs import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "datos" / "creadores.csv"
ARTIFACTS = ("run.json", "summary.json", "bmus.csv", "neurons.csv", "weights.csv", "reconstructed_weights.csv", "hit_map.png", "per_seed.csv", "history_repeats.csv")


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _config() -> dict:
    return json.loads((PROJECT_ROOT / "configs" / "som.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def som_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("som_run") / "runs"
    config = _config()
    config["params"]["n_repeats"] = 2
    path = out.parent / "som.json"
    path.write_text(json.dumps(config))
    assert main(["run", str(path), "--out-root", str(out)]) == 0
    return out / "som"


def test_cli_end_to_end_writes_every_artifact(som_run):
    for name in ARTIFACTS:
        assert (som_run / name).stat().st_size > 0, name


def test_bmus_has_every_creator_exactly_once(som_run):
    creators = _read(DATA_PATH)
    bmus = _read(som_run / "bmus.csv")
    assert len(bmus) == len(creators)
    assert [r["creator_id"] for r in bmus] == [r["creator_id"] for r in creators]
    assert [r["name"] for r in bmus] == [r["name"] for r in creators]
    assert list(bmus[0]) == [
        "creator_id", "name", "dominant_topic", "bmu", "row", "col", "bmu_distance", "bmu_distance_original", "pc1", "pc2",
    ]


def test_neurons_hits_cover_all_creators(som_run):
    cfg = _config()["params"]
    rows = cfg["rows"]
    cols = cfg["cols"] if cfg["cols"] is not None else rows
    neurons = _read(som_run / "neurons.csv")
    assert len(neurons) == rows * cols
    assert sum(int(r["hits"]) for r in neurons) == len(_read(DATA_PATH))


def test_each_name_is_labeled_in_exactly_one_neuron(som_run):
    creators = _read(DATA_PATH)
    bmus = _read(som_run / "bmus.csv")
    n_neurons = len(_read(som_run / "neurons.csv"))
    labels = labels_by_neuron([int(r["bmu"]) for r in bmus], [r["name"] for r in bmus], n_neurons)
    flat = [name for cell in labels for name in cell]
    assert sorted(flat) == sorted(r["name"] for r in creators)
    neurons = _read(som_run / "neurons.csv")
    joined = [name for r in neurons for name in (r["creators"].split(";") if r["creators"] else [])]
    assert sorted(joined) == sorted(r["name"] for r in creators)


def test_summary_counts(som_run):
    summary = json.loads((som_run / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_creators_mapped"] == len(_read(DATA_PATH))
    assert summary["n_samples"] == summary["n_creators_mapped"]
    assert summary["input"] == "pca"
    assert summary["n_original_features"] == 27
    assert summary["n_features"] == summary["n_components"] == 10
    assert summary["retained_variance"] >= 0.9
    assert summary["topology"] == "hex"
    records = _read(som_run / "per_seed.csv")
    assert summary["n_repeats"] == 2
    assert summary["seeds"] == [42, 43]
    assert [int(r["seed"]) for r in records] == summary["seeds"]
    for metric in ("mean_bmu_distance", "mean_bmu_distance_original", "empty_neuron_frac", "u_mean"):
        values = np.array([float(r[metric]) for r in records])
        assert summary[f"{metric}_mean"] == pytest.approx(values.mean(), abs=1e-8)
        assert summary[f"{metric}_std"] == pytest.approx(values.std(ddof=1), abs=1e-8)
    assert summary["mean_bmu_distance_std"] > 0
    assert isinstance(summary["mean_bmu_distance"], float)


def test_same_seed_gives_byte_identical_tables(tmp_path, write_config):
    cfg = write_config(base="configs/som.json", **{"plots": False, "params.n_repeats": 2})
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert main(["run", str(cfg), "--out-root", str(out_a)]) == 0
    assert main(["run", str(cfg), "--out-root", str(out_b)]) == 0
    for name in ("bmus.csv", "weights.csv"):
        assert (out_a / "som" / name).read_bytes() == (out_b / "som" / name).read_bytes()


@pytest.mark.parametrize("overrides", [{"params.rows": 0}, {"params.rows": 1, "params.cols": 1}])
def test_invalid_map_size_exits_2_and_creates_nothing(tmp_path, write_config, capsys, overrides):
    cfg = write_config(base="configs/som.json", **overrides)
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not (out / "som").exists()


INVALID_CASES = [
    {"params.rows": 0},
    {"params.rows": 1, "params.cols": 1},
    {"params.rows": 21, "params.cols": 20},
    {"params.epochs": 0},
    {"params.epochs": 5001},
    {"params.eta0": 0},
    {"params.eta0": 1.5},
    {"params.radius0": 0},
    {"params.topology": "tri"},
    {"params.init": "pca"},
    {"params.radius_decay": "fast"},
    {"params.unknown": 1},
    {"params.rows": True},
    {"params.n_repeats": 1},
    {"params.n_repeats": 101},
    {"params.n_repeats": True},
    {"params.variance_threshold": 0},
    {"params.variance_threshold": 1.1},
    {"params.variance_threshold": True},
    {"params.variance_threshold": 0.1},
    {"pca_config": "configs/som.json"},
    {"pca_config": ""},
]

@pytest.mark.parametrize("overrides", INVALID_CASES, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()))
def test_invalid_som_config_exits_2_and_creates_no_run_dir(tmp_path, write_config, capsys, overrides):
    cfg = write_config(base="configs/som.json", **overrides)
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not (out / "som").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"params.topology": "rect"},
        {"params.init": "random"},
        {"params.radius_decay": "inverse"},
        {"params.rows": 1, "params.cols": 6},
        {"params.rows": 4, "params.cols": 9},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_valid_variants_run(tmp_path, write_config, overrides):
    cfg = write_config(base="configs/som.json", plots=False, **{"params.n_repeats": 2, **overrides})
    assert main(["run", str(cfg), "--out-root", str(tmp_path / "out")]) == 0
    summary = json.loads((tmp_path / "out" / "som" / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_creators_mapped"] == len(_read(DATA_PATH))


def test_history_has_one_row_per_epoch(som_run):
    rows = _read(som_run / "history.csv")
    assert list(rows[0]) == ["epoch", "eta", "radius", "mean_bmu_distance"]
    assert len(rows) == _config()["params"]["epochs"]
    assert [int(r["epoch"]) for r in rows] == list(range(1, len(rows) + 1))


def test_summary_echoes_hyperparameters_and_new_metrics(som_run):
    summary = json.loads((som_run / "summary.json").read_text(encoding="utf-8"))
    p = _config()["params"]
    for key in ("topology", "init", "radius_decay"):
        assert summary[key] == p[key]
    assert summary["mean_bmu_distance"] >= 0
    occupied = summary["n_neurons"] - summary["n_empty_neurons"]
    assert summary["mean_hits_occupied"] == pytest.approx(summary["n_samples"] / occupied, abs=1e-4)


def test_plots_include_the_training_curve(som_run):
    assert (som_run / "training_curve.png").stat().st_size > 0


def test_same_seed_gives_byte_identical_summary_bmus_weights_history(tmp_path, write_config):
    cfg = write_config(base="configs/som.json", plots=False, **{"params.n_repeats": 2})
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert main(["run", str(cfg), "--out-root", str(out_a)]) == 0
    assert main(["run", str(cfg), "--out-root", str(out_b)]) == 0
    for name in ("summary.json", "bmus.csv", "weights.csv", "history.csv"):
        assert (out_a / "som" / name).read_bytes() == (out_b / "som" / name).read_bytes(), name
