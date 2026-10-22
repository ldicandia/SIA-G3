"""QR recall: original data, Hebbian limitations, noise, energy and decoder output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tp4.analysis import qr as qrmod
from tp4.analysis.qr import load_qr_image
from tp4.cli import main
from tp4.config import load_config
from tp4.errors import ConfigError
from tp4.models.hopfield import energy, hebbian_weights
from tp4.runs import PROJECT_ROOT
from tp4.runner import validate_config

N = 37 * 37
REDUCED = {
    "params.n_repeats": 2,
    "params.trials": 4,
    "params.noise_levels": [0.0, 0.1, 0.3, 0.5],
    "params.trace_levels": [0.3],
}


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary(path):
    return json.loads((path / "summary.json").read_text())


def _make_config(directory, name, **overrides):
    config = json.loads((PROJECT_ROOT / "configs/hopfield_qr.json").read_text())
    for dotted, value in overrides.items():
        node = config
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    path = directory / f"{name}.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture(scope="module")
def reduced_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("hopfield_qr")
    cfg = _make_config(directory, "cfg", **REDUCED)
    assert main(["run", str(cfg), "--out-root", str(directory / "runs")]) == 0
    return directory / "runs/hopfield_qr"


def test_reduced_run_writes_every_artifact(reduced_run):
    for name in (
        "stored.csv", "overlaps.csv", "queries.csv", "energy.csv", "energy_trace.csv",
        "storage_control.csv", "single_pattern_control.csv",
        "recall_by_seed.csv", "recall_repetitions.csv", "final_energy.png",
        "recovery_rates.csv", "patterns.png", "overlaps.png", "shared_modules.png",
        "attractor.png", "damage_examples.png", "recovery_vs_noise.png", "iterations.png",
        "energy_per_step.png", "summary.json", "run.json",
    ):
        assert (reduced_run / name).stat().st_size > 0


def test_roundtrip_and_shape_of_the_stored_codes(reduced_run):
    summary = _summary(reduced_run)
    assert summary["roundtrip_identical"]
    assert summary["n_patterns"] == 5 and summary["n_neurons"] == N
    assert summary["side"] == 37 and summary["capacity_015"] == 0.15 * N


def test_full_run_overlap_and_shared_module_facts(reduced_run):
    summary = _summary(reduced_run)
    config = json.loads((PROJECT_ROOT / "configs/hopfield_qr.json").read_text())
    patterns = np.stack([load_qr_image(path).matrix.ravel() for path in config["images"]])
    expected = patterns.astype(float) @ patterns.T / N
    overlaps = _read_csv(reduced_run / "overlaps.csv")
    names = summary["pattern_names"]
    actual = np.array([[float(row[name]) for name in names] for row in overlaps])
    assert np.allclose(actual, expected, atol=1e-6)
    assert summary["mean_offdiag_overlap"] == 0.6701
    assert summary["max_offdiag_overlap"] == 0.8598
    assert summary["n_shared_modules"] == 955
    assert summary["n_fixed_points"] == 0
    assert summary["unstable_bits"] == [71, 91, 225, 225, 103]
    assert summary["spurious_found"]


def test_query_row_count_and_columns(reduced_run):
    rows = _read_csv(reduced_run / "queries.csv")
    assert len(rows) == 5 * 4 * 4 == _summary(reduced_run)["example_n_queries"]
    for row in rows:
        assert int(row["n_changed"]) == round(float(row["level"]) * N)
        assert float(row["damaged_frac"]) == pytest.approx(int(row["n_changed"]) / N, abs=1e-6)
        assert (row["recovered"] == "True") == (int(row["hamming_target"]) == 0)
        assert (row["recovered"] == "True") == (row["final_class"] == "stored" and row["final_pattern"] == row["pattern"])
    zero = [row for row in rows if float(row["level"]) == 0]
    assert len(zero) == 20 and all(row["recovered"] == "False" for row in zero)


def test_energy_is_strictly_decreasing_per_query(reduced_run):
    queries = {row["query_id"]: row for row in _read_csv(reduced_run / "queries.csv")}
    sequences = {}
    for row in _read_csv(reduced_run / "energy.csv"):
        sequences.setdefault(row["query_id"], []).append(float(row["energy"]))
    assert set(sequences) == set(queries)
    for query_id, values in sequences.items():
        assert len(values) == int(queries[query_id]["iterations"]) + 1
        assert all(b < a for a, b in zip(values, values[1:]))
        assert values[-1] == pytest.approx(float(queries[query_id]["final_energy"]), abs=1e-5)
    config = json.loads((PROJECT_ROOT / "configs/hopfield_qr.json").read_text())
    patterns = np.stack([load_qr_image(path).matrix.ravel() for path in config["images"]])
    initial = energy(hebbian_weights(patterns), patterns)
    for row in queries.values():
        if float(row["level"]) == 0:
            index = _summary(reduced_run)["pattern_names"].index(row["pattern"])
            assert sequences[row["query_id"]][0] == pytest.approx(initial[index], abs=1e-5)
    assert _summary(reduced_run)["energy_never_increased"]


def test_energy_trace_has_one_row_per_neuron_visit(reduced_run):
    rows = _read_csv(reduced_run / "energy_trace.csv")
    assert {float(row["level"]) for row in rows} == {0.3}
    values = [float(row["energy"]) for row in rows]
    assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))
    query = next(row for row in _read_csv(reduced_run / "queries.csv") if row["query_id"] == rows[0]["query_id"])
    assert len(rows) == 1 + N * (int(query["iterations"]) + 1)
    assert values[-1] == pytest.approx(float(query["final_energy"]), abs=1e-5)


def test_recovery_rates_table_matches_queries(reduced_run):
    rows = _read_csv(reduced_run / "queries.csv")
    rates = _read_csv(reduced_run / "recovery_rates.csv")
    assert len(rates) == 4
    for rate in rates:
        group = [row for row in rows if row["level"] == rate["level"]]
        assert int(rate["n_queries"]) == len(group)
        assert float(rate["recovery_rate"]) == sum(row["recovered"] == "True" for row in group) / len(group)
        assert float(rate["mean_iterations"]) == pytest.approx(np.mean([int(row["iterations"]) for row in group]), abs=1e-6)


def test_rerun_is_byte_identical(tmp_path):
    cfg = _make_config(tmp_path, "cfg", plots=False, **REDUCED)
    for name in ("a", "b"):
        assert main(["run", str(cfg), "--out-root", str(tmp_path / name)]) == 0
    for name in ("queries.csv", "energy.csv", "energy_trace.csv", "recovery_rates.csv", "recall_by_seed.csv", "recall_repetitions.csv", "summary.json"):
        assert (tmp_path / "a/hopfield_qr" / name).read_bytes() == (tmp_path / "b/hopfield_qr" / name).read_bytes()


def test_decoder_unavailable_keeps_recall_results(tmp_path, monkeypatch):
    monkeypatch.setattr(qrmod, "decoder_status", lambda: qrmod.DecoderStatus(False, "zxing-cpp", None, "unavailable"))
    cfg = _make_config(tmp_path, "cfg", plots=False, **REDUCED)
    assert main(["run", str(cfg), "--out-root", str(tmp_path / "runs")]) == 0
    summary = _summary(tmp_path / "runs/hopfield_qr")
    assert summary["decoder_available"] is False
    assert summary["final_decode_rates"] == []
    assert summary["recovery"] == [0.0] * 4


@pytest.mark.skipif(not qrmod.decoder_status().available, reason="optional zxing-cpp is unavailable")
def test_decode_compares_target_url_and_original_baseline(reduced_run):
    summary = _summary(reduced_run)
    assert summary["n_originals_decoded"] == 5
    assert all(text.startswith("https://") for text in summary["reference_texts"])
    assert summary["query_decode_rates"][0] == 1.0
    assert summary["final_decode_rates"] == [0.0] * 4


@pytest.mark.parametrize("overrides", [
    {"params.unknown": 1}, {"params.n_repeats": 1},
    {"params.trials": 0},
    {"params.noise_levels": []}, {"params.noise_levels": [True]},
    {"params.noise_levels": [0.5, 0.1]}, {"params.noise_levels": [-0.1]},
    {"params.noise_levels": [float("inf")]}, {"params.trace_levels": [0.23]},
    {"params.trace_levels": [0.1, 0.1]}, {"params.update": "sync"},
    {"params.max_iter": 0}, {"params.decode": 1}, {"params.decode_scale": 0},
    {"params.decode_quiet_zone": -1}, {"images": []},
])
def test_invalid_config(tmp_path, overrides):
    cfg = _make_config(tmp_path, "cfg", **overrides)
    with pytest.raises(ConfigError):
        validate_config(load_config(cfg))


def test_storing_individual_qr_codes_and_pairs_preserves_the_patterns(reduced_run):
    rows = _read_csv(reduced_run / "storage_control.csv")
    assert len(rows) == 31
    for row in rows:
        count = int(row["n_stored"])
        assert int(row["n_fixed_points"]) == (count if count <= 2 else 0)
    assert _summary(reduced_run)["singleton_all_fixed"]


def test_individual_qr_networks_recover_noisy_queries(reduced_run):
    rows = _read_csv(reduced_run / "single_pattern_control.csv")
    assert len(rows) == 5 * 4
    assert {row["pattern"] for row in rows} == set(_summary(reduced_run)["pattern_names"])
    for row in rows:
        assert int(row["n_flips"]) == round(float(row["level"]) * N)
        assert float(row["actual_noise"]) == pytest.approx(int(row["n_flips"]) / N, abs=1e-8)
        assert float(row["recovery_rate"]) == 1.0
    assert _summary(reduced_run)["single_pattern_recovery"] == [1.0] * 4


def test_repeated_qr_metrics_report_sample_spread_between_seeds(reduced_run):
    summary = _summary(reduced_run)
    assert summary["n_repeats"] == 2 and summary["seeds"] == [42, 43]
    assert summary["n_queries"] == 160 and summary["n_queries_per_seed"] == 80
    observations = _read_csv(reduced_run / "recall_by_seed.csv")
    repeated = _read_csv(reduced_run / "recall_repetitions.csv")
    for row in repeated:
        group = [r for r in observations if all(r[key] == row[key] for key in ("set", "update", "level"))]
        assert len(group) == 2 and all(int(r["n_queries"]) == 20 for r in group)
        for metric in ("recovery_rate", "mean_iterations", "mean_final_energy"):
            values = np.array([float(r[metric]) for r in group])
            assert float(row[f"{metric}_mean"]) == pytest.approx(values.mean())
            assert float(row[f"{metric}_std"]) == pytest.approx(values.std(ddof=1), abs=1e-7)
    assert summary["recovery_std"] == [0.0] * 4
