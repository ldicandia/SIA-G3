"""End-to-end, failure-evidence, energy, ordering, determinism and validation tests for `hopfield_qr`."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tp4 import runner
from tp4.analysis import qr as qrmod
from tp4.analysis.qr import load_qr_image
from tp4.analysis.qr_damage import CORNERS, corner_distance
from tp4.cli import main
from tp4.config import load_config
from tp4.runs import PROJECT_ROOT

IMAGES = [
    "datos/hopfield/food_menu_qr.png",
    "datos/hopfield/wine_menu_qr.png",
    "datos/hopfield/beverages_menu_qr.png",
    "datos/hopfield/dessert_menu_qr.png",
    "datos/hopfield/specials_menu_qr.png",
]
NAMES = ["food", "wine", "beverages", "dessert", "specials"]
N = 37 * 37

REDUCED = {
    "params.trials": 4,
    "params.noise_levels": [0.0, 0.05, 0.3, 0.45, 1.0],
    "params.torn_corner_sizes": [8, 20],
    "params.occlusion_sizes": [12],
    "params.trace_levels": [0.3],
}

ARTIFACTS = (
    "run.json",
    "summary.json",
    "stored.csv",
    "overlaps.csv",
    "queries.csv",
    "energy.csv",
    "energy_trace.csv",
    "recovery_rates.csv",
    "patterns.png",
    "overlaps.png",
    "shared_modules.png",
    "hebb_attractor.png",
    "damage_examples.png",
    "recovery_vs_noise.png",
    "iterations.png",
    "energy_per_step.png",
)

DAMAGE_ORDER = {"flip": 0, "torn_corner": 1, "occlusion": 2}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _make_config(directory: Path, name: str, **overrides) -> Path:
    config = json.loads((PROJECT_ROOT / "configs" / "hopfield_qr.json").read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        node = config
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    path = directory / f"{name}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def reduced_run(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("hopfield_qr_reduced")
    cfg = _make_config(base, "cfg", **REDUCED)
    assert main(["run", str(cfg), "--out-root", str(base / "runs")]) == 0
    return base / "runs" / "hopfield_qr"


@pytest.fixture(scope="module")
def full_run(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("hopfield_qr_full")
    cfg = _make_config(base, "cfg", plots=False)
    assert main(["run", str(cfg), "--out-root", str(base / "runs")]) == 0
    return base / "runs" / "hopfield_qr"


# --------------------------------------------------------------------------- reduced run


def test_reduced_run_writes_every_artifact(reduced_run: Path):
    for name in ARTIFACTS:
        p = reduced_run / name
        assert p.is_file() and p.stat().st_size > 0, f"missing or empty {name}"


def test_roundtrip_and_shape_of_the_stored_codes(reduced_run: Path):
    s = _summary(reduced_run)
    assert s["roundtrip_identical"] is True
    assert s["n_patterns"] == 5 and s["n_neurons"] == N and s["side"] == 37
    assert s["pattern_names"] == NAMES
    assert s["capacity_0138"] == 188.922 and s["capacity_015"] == 205.35


def test_query_row_count_and_columns(reduced_run: Path, csv_rows):
    rows = _read_csv(reduced_run / "queries.csv")
    assert len(rows) == csv_rows(reduced_run / "queries.csv") == 5 * (5 * 4 + 2 * 4 + 1 * 4) * 2
    header = list(rows[0].keys())
    assert header[:14] == [
        "query_id", "damage", "level", "size", "corner", "top", "left", "pattern", "trial",
        "n_changed", "damaged_frac", "rule", "iterations", "status",
    ]  # fmt: skip
    assert {r["damage"] for r in rows} == {"flip", "torn_corner", "occlusion"}
    s = _summary(reduced_run)
    assert s["n_queries"] == len(rows) // 2 and s["n_recalls"] == len(rows)


def test_recovered_means_exact_equality_with_the_target_on_every_row(reduced_run: Path):
    for r in _read_csv(reduced_run / "queries.csv"):
        exact = r["final_class"] == "stored" and r["final_pattern"] == r["pattern"]
        assert (r["recovered"] == "True") == exact, r
        assert (r["recovered"] == "True") == (int(r["hamming_target"]) == 0), r
        if r["recovered"] == "True":
            assert float(r["overlap_target"]) == 1.0


def test_level_zero_and_one_boundaries(reduced_run: Path):
    rows = _read_csv(reduced_run / "queries.csv")
    zero = [r for r in rows if r["damage"] == "flip" and float(r["level"]) == 0.0]
    assert len(zero) == 5 * 4 * 2
    for r in zero:
        assert r["n_changed"] == "0" and r["damaged_frac"] == "0"
        if r["rule"] == "pseudoinverse":
            assert r["recovered"] == "True" and r["iterations"] == "0" and r["status"] == "converged"
        else:
            assert r["recovered"] == "False"
    one = [r for r in rows if r["damage"] == "flip" and float(r["level"]) == 1.0 and r["rule"] == "pseudoinverse"]
    assert len(one) == 5 * 4
    for r in one:
        assert r["n_changed"] == str(N) and r["damaged_frac"] == "1"
        assert r["final_class"] == "inverted" and r["iterations"] == "0" and r["recovered"] == "False"


def test_hebb_always_ends_in_the_average_qr_and_pseudoinverse_recovers_corners(reduced_run: Path):
    rows = _read_csv(reduced_run / "queries.csv")
    hebb = [r for r in rows if r["rule"] == "hebb"]
    assert not any(r["recovered"] == "True" for r in hebb)
    for r in hebb:
        if r["damage"] != "flip" or float(r["level"]) <= 0.45:
            assert r["is_mean_attractor"] == "True", r
            assert r["final_class"] == "spurious"
    small_corner = [r for r in rows if r["rule"] == "pseudoinverse" and r["damage"] == "torn_corner" and r["size"] == "8"]
    assert len(small_corner) == 5 * 4
    assert all(r["recovered"] == "True" for r in small_corner)


def test_damage_masks_match_the_originals(reduced_run: Path):
    originals = {n: load_qr_image(p).matrix for n, p in zip(NAMES, IMAGES, strict=True)}
    rows = [r for r in _read_csv(reduced_run / "queries.csv") if r["rule"] == "hebb"]
    n_torn = n_occ = 0
    for r in rows:
        m = originals[r["pattern"]]
        if r["damage"] == "torn_corner":
            n_torn += 1
            tri = corner_distance(37, r["corner"]) < int(r["size"])
            assert int(r["n_changed"]) == int(np.sum(m[tri] == 1))
            assert r["level"] == "" and r["top"] == "" and r["trial"] == "0"
        elif r["damage"] == "occlusion":
            n_occ += 1
            top, left, size = int(r["top"]), int(r["left"]), int(r["size"])
            assert 0 <= top <= 37 - size and 0 <= left <= 37 - size
            assert int(r["n_changed"]) == int(np.sum(m[top : top + size, left : left + size] == -1))
            assert r["corner"] == "" and r["level"] == ""
        else:
            assert r["size"] == "" and r["corner"] == "" and r["top"] == ""
            assert int(r["n_changed"]) == round(float(r["level"]) * N)
    assert n_torn == 5 * 2 * 4 and n_occ == 5 * 4


def test_flip_count_uses_round_half_to_even(reduced_run: Path):
    rows = [r for r in _read_csv(reduced_run / "queries.csv") if r["damage"] == "flip" and r["rule"] == "hebb"]
    by_level = {float(r["level"]): int(r["n_changed"]) for r in rows}
    assert by_level == {0.0: 0, 0.05: 68, 0.3: 411, 0.45: 616, 1.0: N}
    assert round(0.15 * N) == 205


def test_query_ids_are_shared_by_the_rules_and_ordered(reduced_run: Path):
    rows = _read_csv(reduced_run / "queries.csv")
    ids = [int(r["query_id"]) for r in rows]
    assert ids == sorted(ids)
    assert [r["rule"] for r in rows[:4]] == ["hebb", "pseudoinverse", "hebb", "pseudoinverse"]
    assert ids[0] == ids[1] != ids[2]
    assert ids[-1] == len(rows) // 2 - 1
    previous = None
    for r in rows:
        if r["rule"] != "hebb":
            continue
        pattern = NAMES.index(r["pattern"])
        if r["damage"] == "flip":
            key = (0, float(r["level"]), pattern, 0, int(r["trial"]))
        elif r["damage"] == "torn_corner":
            key = (1, int(r["size"]), pattern, CORNERS.index(r["corner"]), int(r["trial"]))
        else:
            key = (2, int(r["size"]), pattern, 0, int(r["trial"]))
        assert previous is None or key > previous, (previous, key)
        previous = key
    energy_rows = _read_csv(reduced_run / "energy.csv")
    order = [(int(r["query_id"]), 0 if r["rule"] == "hebb" else 1, int(r["iteration"])) for r in energy_rows]
    assert order == sorted(order)


def test_energy_is_strictly_decreasing_per_query_and_rule(reduced_run: Path):
    seqs: dict[tuple[str, str], list[float]] = {}
    for r in _read_csv(reduced_run / "energy.csv"):
        seqs.setdefault((r["query_id"], r["rule"]), []).append(float(r["energy"]))
    queries = {(r["query_id"], r["rule"]): r for r in _read_csv(reduced_run / "queries.csv")}
    assert set(seqs) == set(queries)
    for key, seq in seqs.items():
        assert len(seq) == int(queries[key]["iterations"]) + 1
        assert all(b < a for a, b in zip(seq, seq[1:], strict=False)), key
        assert seq[-1] == pytest.approx(float(queries[key]["final_energy"]), abs=1e-5)
    assert _summary(reduced_run)["energy_never_increased"] is True


def test_energy_trace_has_one_row_per_neuron_visit(reduced_run: Path):
    rows = _read_csv(reduced_run / "energy_trace.csv")
    assert list(rows[0].keys()) == ["rule", "level", "pattern", "step", "sweep", "energy"]
    assert {(r["rule"], r["level"], r["pattern"]) for r in rows} == {("hebb", "0.3", "food"), ("pseudoinverse", "0.3", "food")}
    queries = {
        r["rule"]: r
        for r in _read_csv(reduced_run / "queries.csv")
        if r["damage"] == "flip" and r["level"] == "0.3" and r["pattern"] == "food" and r["trial"] == "0"
    }
    for rule, query in queries.items():
        trace = [r for r in rows if r["rule"] == rule]
        values = [float(r["energy"]) for r in trace]
        assert [int(r["step"]) for r in trace] == list(range(len(trace)))
        assert len(trace) == 1 + N * (int(query["iterations"]) + 1)
        assert [int(r["sweep"]) for r in trace][0] == 0 and int(trace[-1]["sweep"]) == int(query["iterations"]) + 1
        assert all(b <= a + 1e-9 for a, b in zip(values, values[1:], strict=False))
        assert any(b == a for a, b in zip(values, values[1:], strict=False)), "updates that do not flip keep the energy"
        assert values[-1] == pytest.approx(float(query["final_energy"]), abs=1e-5)


def test_recovery_rates_table_matches_queries(reduced_run: Path):
    rates = _read_csv(reduced_run / "recovery_rates.csv")
    assert len(rates) == (5 + 2 + 1) * 2
    assert list(rates[0].keys()) == [
        "damage", "level", "size", "rule", "n_queries", "mean_damaged_frac", "recovery_rate", "other_rate",
        "inverted_rate", "spurious_rate", "mean_attractor_rate", "max_iter_rate", "mean_iterations",
        "max_iterations", "mean_overlap_target",
    ]  # fmt: skip
    queries = _read_csv(reduced_run / "queries.csv")
    for r in rates:
        sub = [
            q
            for q in queries
            if q["damage"] == r["damage"] and q["rule"] == r["rule"]
            and (q["level"] == r["level"] if r["damage"] == "flip" else q["size"] == r["size"])
        ]  # fmt: skip
        assert int(r["n_queries"]) == len(sub)
        assert float(r["recovery_rate"]) == round(sum(q["recovered"] == "True" for q in sub) / len(sub), 4)
        assert float(r["max_iterations"]) == max(int(q["iterations"]) for q in sub)
        total = sum(float(r[k]) for k in ("recovery_rate", "other_rate", "inverted_rate", "spurious_rate"))
        assert total == pytest.approx(1.0, abs=2e-4)


def test_overlaps_and_shared_modules_in_the_reduced_run(reduced_run: Path):
    s = _summary(reduced_run)
    rows = _read_csv(reduced_run / "overlaps.csv")
    assert [r["pattern"] for r in rows] == NAMES
    assert list(rows[0].keys()) == ["pattern", *NAMES]
    assert all(float(rows[i][NAMES[i]]) == 1.0 for i in range(5))
    assert float(rows[0]["wine"]) == pytest.approx(s["max_offdiag_overlap"], abs=1e-4)
    assert s["max_overlap_pair"] == "food-wine"
    assert len(s["hebb_mean_attractor_overlaps"]) == 5
    assert s["corners"] == list(CORNERS)
    assert s["torn_corner_sizes"] == [8, 20] and s["occlusion_sizes"] == [12]
    assert len(s["hebb_corner_recovery"]) == 2 and len(s["pseudoinverse_occlusion_recovery"]) == 1


def test_rerun_is_byte_identical(tmp_path: Path):
    cfg = _make_config(tmp_path, "cfg", plots=False, **REDUCED)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert main(["run", str(cfg), "--out-root", str(out_a)]) == 0
    assert main(["run", str(cfg), "--out-root", str(out_b)]) == 0
    for name in ("queries.csv", "energy.csv", "energy_trace.csv", "recovery_rates.csv", "stored.csv", "overlaps.csv", "summary.json"):
        assert (out_a / "hopfield_qr" / name).read_bytes() == (out_b / "hopfield_qr" / name).read_bytes(), name


def test_max_iter_boundary(tmp_path: Path):
    cfg = _make_config(tmp_path, "cfg", plots=False, **{**REDUCED, "params.max_iter": 1})
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    rows = _read_csv(out / "hopfield_qr" / "queries.csv")
    capped = [r for r in rows if r["status"] == "max_iter"]
    assert capped and all(r["iterations"] == "1" for r in capped)
    assert any(r["damage"] == "flip" and r["level"] == "0.45" for r in capped)
    assert all(r["iterations"] == "0" for r in rows if r["status"] == "converged" and r["rule"] == "pseudoinverse" and r["level"] == "0")
    s = _summary(out / "hopfield_qr")
    assert s["n_max_iter"] == len(capped) and s["max_iterations"] == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"params.corners": [], "params.torn_corner_sizes": [], "params.occlusion_sizes": [], "params.trace_levels": []},
        {"params.corners": []},
        {"params.torn_corner_sizes": []},
        {"params.occlusion_sizes": []},
        {"params.trace_levels": []},
    ],
    ids=lambda o: ",".join(f"{k.split('.')[-1]}={v}" for k, v in o.items()),
)
def test_empty_damage_lists_write_no_rows_and_still_exit_0(tmp_path: Path, overrides):
    cfg = _make_config(tmp_path, "cfg", **{**REDUCED, **overrides})  # plots on: empty panels must still draw
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    run_dir = out / "hopfield_qr"
    rows = _read_csv(run_dir / "queries.csv")
    damage = {r["damage"] for r in rows}
    s = _summary(run_dir)
    no_corner = overrides.get("params.corners") == [] or overrides.get("params.torn_corner_sizes") == []
    no_stain = overrides.get("params.occlusion_sizes") == []
    assert ("torn_corner" not in damage) == no_corner
    assert ("occlusion" not in damage) == no_stain
    assert "flip" in damage
    if no_corner:
        assert s["hebb_corner_recovery"] == [] and s["pseudoinverse_corner_mean_damaged_frac"] == []
    if no_stain:
        assert s["hebb_occlusion_recovery"] == [] and s["pseudoinverse_occlusion_mean_damaged_frac"] == []
    trace = _read_csv(run_dir / "energy_trace.csv")
    assert (len(trace) == 0) == (overrides.get("params.trace_levels") == [])
    for name in ARTIFACTS:
        assert (run_dir / name).is_file(), name


# --------------------------------------------------------------------------- full config facts


def test_full_run_overlap_and_shared_module_facts(full_run: Path):
    s = _summary(full_run)
    assert s["max_offdiag_overlap"] == 0.8598 and s["max_overlap_pair"] == "food-wine"
    assert s["mean_offdiag_overlap"] == 0.6701
    assert s["n_shared_modules"] == 955 and s["shared_module_frac"] == 0.6976
    assert s["hebb_n_fixed_points"] == 0 and s["pseudoinverse_n_fixed_points"] == 5
    assert s["hebb_unstable_bits"] == [71, 91, 225, 225, 103]
    assert s["pseudoinverse_unstable_bits"] == [0, 0, 0, 0, 0]
    assert s["capacity_0138"] == 188.922 and s["n_patterns"] == 5


def test_full_run_hebb_fails_and_pseudoinverse_recovers(full_run: Path, csv_rows):
    s = _summary(full_run)
    assert csv_rows(full_run / "queries.csv") == 5 * (11 * 20 + 4 * 7 + 5 * 20) * 2 == 3480
    assert {r["damage"] for r in _read_csv(full_run / "queries.csv")} == {"flip", "torn_corner", "occlusion"}
    assert s["hebb_frac_mean_attractor"] >= 0.95
    assert max(s["hebb_flip_recovery"]) == 0.0
    assert all(v == 0.0 for v in s["hebb_corner_recovery"]) and len(s["hebb_corner_recovery"]) == 7
    assert all(v == 0.0 for v in s["hebb_occlusion_recovery"]) and len(s["hebb_occlusion_recovery"]) == 5
    assert s["hebb_max_level_full_recovery"] is None
    assert s["pseudoinverse_corner_recovery"][0] == 1.0
    assert s["pseudoinverse_max_level_full_recovery"] == 0.35
    assert s["pseudoinverse_flip_recovery"][:8] == [1.0] * 8
    assert s["pseudoinverse_occlusion_recovery"][:4] == [1.0] * 4  # sizes 8 to 20
    assert min(s["pseudoinverse_occlusion_recovery"]) >= 0.95
    assert s["pseudoinverse_corner_recovery"][:6] == [1.0] * 6  # sizes 8 to 28
    assert s["energy_never_increased"] is True
    assert len(s["hebb_mean_attractor_overlaps"]) == 5
    assert s["hebb_mean_attractor_overlaps"][0] == pytest.approx(0.8963, abs=1e-4)
    trace = _read_csv(full_run / "energy_trace.csv")
    assert {(r["rule"], r["level"]) for r in trace} == {(rule, lv) for rule in ("hebb", "pseudoinverse") for lv in ("0.1", "0.3", "0.45")}


def test_full_run_mean_attractor_overlaps_match_the_codes(full_run: Path):
    P = np.stack([load_qr_image(p).matrix.reshape(-1) for p in IMAGES]).astype(int)
    mean_attractor = np.where(P.sum(axis=0) >= 0, 1, -1)
    expected = [round(float(mean_attractor @ P[k]) / N, 4) for k in range(5)]
    assert _summary(full_run)["hebb_mean_attractor_overlaps"] == expected


# --------------------------------------------------------------------------- validation & boundaries


def _images_with(*extra: str) -> list[str]:
    return [*IMAGES[:2], *extra]


INVALID_CASES = [
    {"images": [IMAGES[0]]},
    {"images": [IMAGES[0], IMAGES[0]]},
    {"images": [IMAGES[0], "datos/hopfield/missing_menu_qr.png"]},
    {"images": [IMAGES[0], "datos/letras_5x5.json"]},
    {"images": "datos/hopfield/food_menu_qr.png"},
    {"images": []},
    {"params.side": 4},
    {"params.side": 65},
    {"params.rules": []},
    {"params.rules": ["hebb", "hebb"]},
    {"params.rules": ["storkey"]},
    {"params.update": "greedy"},
    {"params.max_iter": 0},
    {"params.max_iter": 1001},
    {"params.noise_levels": [], "params.trace_levels": []},
    {"params.noise_levels": [-0.1, 0.2], "params.trace_levels": []},
    {"params.noise_levels": [0.2, 1.1], "params.trace_levels": []},
    {"params.noise_levels": [0.2, 0.2], "params.trace_levels": []},
    {"params.noise_levels": [0.3, 0.2], "params.trace_levels": []},
    {"params.noise_levels": ["a"], "params.trace_levels": []},
    {"params.noise_levels": [i / 100 for i in range(22)], "params.trace_levels": []},
    {"params.trials": 0},
    {"params.trials": 201},
    {"params.corners": ["top_left", "top_left"]},
    {"params.corners": ["middle"]},
    {"params.corners": "top_left"},
    {"params.torn_corner_sizes": [0]},
    {"params.torn_corner_sizes": [38]},
    {"params.torn_corner_sizes": [8, 8]},
    {"params.torn_corner_sizes": [20, 8]},
    {"params.torn_corner_sizes": [8.5]},
    {"params.occlusion_sizes": [0]},
    {"params.occlusion_sizes": [38]},
    {"params.occlusion_sizes": [12, 12]},
    {"params.trace_levels": [0.33]},
    {"params.trace_levels": [0.3, 0.3]},
    {"params.update": "sync"},
    {"params.noise_levels": [i / 100 for i in range(21)], "params.trials": 200, "params.trace_levels": []},
    {"params.unknown": 1},
]


@pytest.mark.parametrize("overrides", INVALID_CASES, ids=lambda o: ",".join(f"{k}={str(v)[:30]}" for k, v in o.items()))
def test_invalid_hopfield_qr_config_exits_2_and_creates_no_run_dir(tmp_path: Path, capsys, overrides):
    cfg = _make_config(tmp_path, "cfg", **overrides)
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not (out / "hopfield_qr").exists()


def test_eleven_images_exit_2(tmp_path: Path, capsys):
    copies = []
    for i in range(11):
        target = tmp_path / f"copy{i}_menu_qr.png"
        shutil.copy(PROJECT_ROOT / IMAGES[0], target)
        copies.append(str(target))
    cfg = _make_config(tmp_path, "cfg", images=copies)
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not (out / "hopfield_qr").exists()


def test_ten_images_pass_validation(tmp_path: Path):
    copies = []
    for i in range(10):
        target = tmp_path / f"copy{i}_menu_qr.png"
        shutil.copy(PROJECT_ROOT / IMAGES[0], target)
        copies.append(str(target))
    cfg = _make_config(tmp_path, "cfg", images=copies, **{"params.trials": 1})
    assert runner.validate_config(load_config(cfg))["kind"] == "hopfield_qr"


def test_wrong_size_and_duplicate_images_exit_2_and_keep_the_previous_run(tmp_path: Path, capsys):
    out = tmp_path / "out"
    good = _make_config(tmp_path, "good", plots=False, **{**REDUCED, "params.trials": 1})
    assert main(["run", str(good), "--out-root", str(out)]) == 0
    summary_before = (out / "hopfield_qr" / "summary.json").read_bytes()
    capsys.readouterr()

    wrong = tmp_path / "wrong_menu_qr.png"
    Image.new("L", (37, 38), 255).save(wrong)
    dup = tmp_path / "food_copy_menu_qr.png"
    shutil.copy(PROJECT_ROOT / IMAGES[0], dup)
    for idx, images in enumerate(([str(wrong), *IMAGES[1:]], [IMAGES[0], str(dup), *IMAGES[2:]])):
        cfg = _make_config(tmp_path, f"bad{idx}", images=images)
        assert main(["run", str(cfg), "--out-root", str(out)]) == 2
        assert "error:" in capsys.readouterr().err
        assert (out / "hopfield_qr" / "summary.json").read_bytes() == summary_before
        assert not list(out.glob(".staging-*"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"params.noise_levels": [0.0, 1.0], "params.trace_levels": [0.0], "params.torn_corner_sizes": [1, 37],
         "params.occlusion_sizes": [37], "params.trials": 200},
        {"params.noise_levels": [0.0], "params.trace_levels": []},
        {"params.max_iter": 1},
        {"params.max_iter": 1000},
        {"params.side": 5, "params.torn_corner_sizes": [5], "params.occlusion_sizes": [5]},
        {"params.noise_levels": [i / 100 for i in range(21)], "params.trace_levels": [], "params.trials": 100},
    ],
    ids=lambda o: ",".join(f"{k}={str(v)[:20]}" for k, v in o.items()),
)  # fmt: skip
def test_boundary_values_pass_validation(tmp_path: Path, overrides):
    cfg = _make_config(tmp_path, "cfg", **overrides)
    config = runner.validate_config(load_config(cfg))
    assert config["kind"] == "hopfield_qr"


# --------------------------------------------------------------------------- decoder (Task 3)

DECODE_COLUMNS = ["query_decode_ok", "final_decode_ok", "final_decoded_as"]


def test_queries_header_ends_with_the_decode_columns(reduced_run: Path):
    header = list(_read_csv(reduced_run / "queries.csv")[0].keys())
    assert header[-4:] == ["final_energy", *DECODE_COLUMNS]
    rates = list(_read_csv(reduced_run / "recovery_rates.csv")[0].keys())
    assert rates[-2:] == ["query_decode_rate", "final_decode_rate"]


def test_decoder_available_reads_originals_queries_and_recovered_states(reduced_run: Path):
    pytest.importorskip("zxingcpp")
    s = _summary(reduced_run)
    assert s["decode_enabled"] is True and s["decoder_available"] is True
    assert s["decoder_name"] == "zxing-cpp" and s["decoder_reason"] is None
    assert s["decode_scale"] == 8 and s["decode_quiet_zone"] == 4
    assert s["n_originals_decoded"] == 5
    texts = s["reference_texts"]
    assert len(texts) == 5 and len(set(texts)) == 5
    for name, text in zip(NAMES, texts, strict=True):
        assert f"{name}_menu" in text

    rows = _read_csv(reduced_run / "queries.csv")
    for r in rows:
        if r["damage"] == "flip" and r["level"] == "0":
            assert r["query_decode_ok"] == "True", r
        if r["recovered"] == "True":
            assert r["final_decode_ok"] == "True" and r["final_decoded_as"] == r["pattern"], r
        if r["final_decode_ok"] == "True":
            assert r["final_decoded_as"] == r["pattern"]
        assert r["final_decoded_as"] in (*NAMES, "other", "none")
        assert r["query_decode_ok"] in ("True", "False") and r["final_decode_ok"] in ("True", "False")
    inverted = [r for r in rows if r["damage"] == "flip" and r["level"] == "1" and r["rule"] == "pseudoinverse"]
    assert inverted and all(r["query_decode_ok"] == "False" for r in inverted)

    rates = _read_csv(reduced_run / "recovery_rates.csv")
    assert all(r["query_decode_rate"] != "" and r["final_decode_rate"] != "" for r in rates)
    assert (reduced_run / "decode_vs_noise.png").stat().st_size > 0
    assert s["pseudoinverse_final_flip_decode_rates"][0] == 1.0
    assert len(s["query_flip_decode_rates"]) == 5 and len(s["pseudoinverse_final_corner_decode_rates"]) == 2
    assert len(s["query_occlusion_decode_rates"]) == 1 and len(s["hebb_final_occlusion_decode_rates"]) == 1
    assert s["query_max_level_full_decode"] is not None and s["query_max_level_full_decode"] >= 0.0
    assert s["pseudoinverse_max_level_full_decode"] is not None
    assert s["mean_attractor_decoded_as"] in (*NAMES, "other", "none")


def test_reference_texts_equal_a_fresh_decode_of_each_original(reduced_run: Path):
    pytest.importorskip("zxingcpp")
    texts = _summary(reduced_run)["reference_texts"]
    for name, path, text in zip(NAMES, IMAGES, texts, strict=True):
        assert qrmod.decode_matrix(load_qr_image(path).matrix, scale=8, quiet_zone=4) == text, name


def _unavailable(monkeypatch):
    status = qrmod.DecoderStatus(False, "zxing-cpp", None, "ModuleNotFoundError: test")
    monkeypatch.setattr(qrmod, "decoder_status", lambda: status)
    _forbid_decode(monkeypatch)


def _forbid_decode(monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("decode_matrix must not be called")

    monkeypatch.setattr(qrmod, "decode_matrix", boom)


def test_decoder_unavailable_degrades_cleanly(tmp_path: Path, monkeypatch):
    _unavailable(monkeypatch)
    cfg = _make_config(tmp_path, "cfg", **REDUCED)
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    run_dir = out / "hopfield_qr"
    s = _summary(run_dir)
    assert s["decode_enabled"] is True and s["decoder_available"] is False
    assert s["decoder_reason"] == "ModuleNotFoundError: test"
    assert s["reference_texts"] == [] and s["n_originals_decoded"] is None
    assert s["query_flip_decode_rates"] == [] and s["pseudoinverse_final_flip_decode_rates"] == []
    assert s["query_max_level_full_decode"] is None and s["mean_attractor_decoded_as"] is None
    for r in _read_csv(run_dir / "queries.csv"):
        assert r["query_decode_ok"] == "" and r["final_decode_ok"] == "" and r["final_decoded_as"] == ""
    for r in _read_csv(run_dir / "recovery_rates.csv"):
        assert r["query_decode_rate"] == "" and r["final_decode_rate"] == ""
    assert not (run_dir / "decode_vs_noise.png").exists()
    for name in ARTIFACTS:
        assert (run_dir / name).is_file(), name


def test_decode_false_never_calls_the_decoder(tmp_path: Path, monkeypatch):
    _forbid_decode(monkeypatch)
    cfg = _make_config(tmp_path, "cfg", **{**REDUCED, "params.decode": False})
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    run_dir = out / "hopfield_qr"
    s = _summary(run_dir)
    assert s["decode_enabled"] is False and s["decoder_available"] in (False, None)
    assert s["reference_texts"] == [] and s["query_flip_decode_rates"] == []
    assert not (run_dir / "decode_vs_noise.png").exists()
    assert all(r["final_decode_ok"] == "" for r in _read_csv(run_dir / "queries.csv"))


def test_decode_bookkeeping_with_a_fake_reader(tmp_path: Path, monkeypatch):
    """A reader that only 'sees' a code when it equals an original exactly."""
    originals = {load_qr_image(p).matrix.reshape(-1).tobytes(): n for n, p in zip(NAMES, IMAGES, strict=True)}

    def fake_decode(matrix, *, scale, quiet_zone):
        name = originals.get(np.asarray(matrix, dtype=np.int8).reshape(-1).tobytes())
        return None if name is None else f"https://x/{name}_menu"

    monkeypatch.setattr(qrmod, "decoder_status", lambda: qrmod.DecoderStatus(True, "fake", "0", None))
    monkeypatch.setattr(qrmod, "decode_matrix", fake_decode)
    cfg = _make_config(tmp_path, "cfg", plots=False, **REDUCED)
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    run_dir = out / "hopfield_qr"
    s = _summary(run_dir)
    assert s["decoder_available"] is True and s["n_originals_decoded"] == 5
    assert s["reference_texts"] == [f"https://x/{n}_menu" for n in NAMES]
    for r in _read_csv(run_dir / "queries.csv"):
        assert (r["query_decode_ok"] == "True") == (r["n_changed"] == "0"), r
        assert (r["final_decode_ok"] == "True") == (r["recovered"] == "True"), r
        if r["final_class"] == "spurious":
            assert r["final_decoded_as"] == "none"
        elif r["final_class"] == "stored":
            assert r["final_decoded_as"] == r["final_pattern"]
    assert s["mean_attractor_decoded_as"] == "none"
    assert not (run_dir / "decode_vs_noise.png").exists()  # plots are off


INVALID_DECODE_CASES = [
    {"params.decode": "yes"},
    {"params.decode": 1},
    {"params.decode_scale": 0},
    {"params.decode_scale": 17},
    {"params.decode_scale": 8.0},
    {"params.decode_quiet_zone": -1},
    {"params.decode_quiet_zone": 11},
]


@pytest.mark.parametrize("overrides", INVALID_DECODE_CASES, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()))
def test_invalid_decode_params_exit_2(tmp_path: Path, capsys, overrides):
    cfg = _make_config(tmp_path, "cfg", **overrides)
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not (out / "hopfield_qr").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"params.decode_scale": 1},
        {"params.decode_scale": 16},
        {"params.decode_quiet_zone": 0},
        {"params.decode_quiet_zone": 10},
        {"params.decode": False},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_decode_boundary_values_pass_validation(tmp_path: Path, overrides):
    cfg = _make_config(tmp_path, "cfg", **overrides)
    assert runner.validate_config(load_config(cfg))["kind"] == "hopfield_qr"
