"""End-to-end, selection, capacity, recall, spurious, determinism and validation tests for `hopfield_letters`."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from tp4 import runner
from tp4.analysis.hopfield_eval import (
    classify_states,
    dot_matrix,
    flip_bits,
    n_flips_for,
    nearest_pattern,
    overlaps,
    rank_subsets,
    render_rows,
    stability_margins,
)
from tp4.cli import main
from tp4.config import load_config
from tp4.experiments.hopfield_letters import load_alphabet
from tp4.models.hopfield import HopfieldConfig, is_fixed_point, recall, hebbian_weights
from tp4.runs import PROJECT_ROOT

ARTIFACTS = (
    "run.json",
    "summary.json",
    "dot_products.csv",
    "orthogonality.csv",
    "stored.csv",
    "recall.csv",
    "recall_rates.csv",
    "energy.csv",
    "energy_trace.csv",
    "capacity.csv",
    "recall_by_seed.csv", "recall_repetitions.csv",
    "capacity_by_seed.csv", "capacity_repetitions.csv", "final_energy.png",
    "alphabet.png",
    "dot_products.png",
    "capacity.png",
    "recovery_vs_noise.png",
    "recall_examples.png",
    "spurious.png",
    "energy.png",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _glyph_to_pattern(rows: list[str]) -> np.ndarray:
    return np.asarray([1 if ch == "#" else -1 for row in rows for ch in row], dtype=np.int8)


@pytest.fixture(scope="module")
def letters_run(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("hopfield_letters_e2e")
    out = base / "runs"
    config = load_config(PROJECT_ROOT / "configs/hopfield_letters.json")
    config["params"]["n_repeats"] = 2
    path = base / "letters.json"
    path.write_text(json.dumps(config))
    assert main(["run", str(path), "--out-root", str(out)]) == 0
    return out / "hopfield_letters"


# --------------------------------------------------------------------------- e2e & selection


def test_cli_end_to_end_writes_every_artifact(letters_run: Path):
    for name in ARTIFACTS:
        p = letters_run / name
        assert p.is_file() and p.stat().st_size > 0, f"missing or empty {name}"


def test_selection_matches_recomputed_argmin_and_orthogonality_ranking(letters_run: Path):
    letters, patterns, shape = load_alphabet("datos/letras_5x5.json")
    assert shape == (5, 5)
    assert len(letters) == 26
    dots = dot_matrix(patterns)

    best_key = None
    worst_key = None
    best_letters = None
    worst_letters = None
    for combo in itertools.combinations(range(26), 4):
        off = [abs(int(dots[a, b])) for idx_a, a in enumerate(combo) for b in combo[idx_a + 1 :]]
        max_abs = max(off)
        mean_abs = sum(off) / len(off)
        lt = tuple(letters[i] for i in combo)
        k_min = (max_abs, mean_abs, lt)
        if best_key is None or k_min < best_key:
            best_key = k_min
            best_letters = list(lt)
        if worst_key is None or k_min > worst_key:
            worst_key = k_min
            worst_letters = list(lt)

    s = _summary(letters_run)
    assert s["selected_letters"] == best_letters == ["G", "R", "T", "V"]
    assert s["selected_max_abs_dot"] == 1
    assert s["selected_mean_abs_dot"] == 1.0
    assert s["worst_letters"] == worst_letters == ["H", "M", "N", "W"]
    assert s["worst_max_abs_dot"] == 21
    assert s["n_subsets"] == 14950
    assert isinstance(s["selection_rule"], str) and "ortogonal" in s["selection_rule"].lower() or "producto" in s["selection_rule"].lower()

    orth = _read_csv(letters_run / "orthogonality.csv")
    assert len(orth) == 14950
    assert orth[0]["rank"] == "1" and orth[0]["letters"] == "GRTV"
    assert int(orth[0]["max_abs_dot"]) == 1 and float(orth[0]["mean_abs_dot"]) == pytest.approx(1.0)
    assert orth[-1]["rank"] == "14950" and orth[-1]["letters"] == "HMNW"
    assert int(orth[-1]["max_abs_dot"]) == 21


def test_stored_table_and_stability_margins(letters_run: Path):
    rows = _read_csv(letters_run / "stored.csv")
    assert len(rows) == 8
    sel = [r for r in rows if r["set"] == "selected"]
    wst = [r for r in rows if r["set"] == "worst"]
    assert [r["letter"] for r in sel] == ["G", "R", "T", "V"]
    assert all(r["fixed_point"] == "True" for r in sel)
    assert all(float(r["stability_margin"]) == pytest.approx(0.72, abs=1e-4) for r in sel)
    assert [r["letter"] for r in wst] == ["H", "M", "N", "W"]
    assert [r["letter"] for r in wst if r["fixed_point"] == "True"] == ["N"]

    s = _summary(letters_run)
    assert s["selected_n_fixed_points"] == 4
    assert s["worst_n_fixed_points"] == 1
    assert s["selected_stability_margins"] == [0.72, 0.72, 0.72, 0.72]


# --------------------------------------------------------------------------- exhaustive 1- and 2-flip recovery


def test_exhaustive_one_and_two_flip_queries_converge_to_target():
    letters, patterns, _ = load_alphabet("datos/letras_5x5.json")
    sel_idx = [letters.index(ch) for ch in ("G", "R", "T", "V")]
    P_sel = patterns[sel_idx]
    W = hebbian_weights(P_sel)
    n = P_sel.shape[1]

    one_flip_positions = [(i,) for i in range(n)]
    two_flip_positions = list(itertools.combinations(range(n), 2))
    all_flips = one_flip_positions + two_flip_positions

    queries_list = []
    targets_list = []
    for pat in P_sel:
        q_pat = np.repeat(pat[np.newaxis, :], len(all_flips), axis=0)
        for row_idx, pos in enumerate(all_flips):
            q_pat[row_idx, list(pos)] = -q_pat[row_idx, list(pos)]
        queries_list.append(q_pat)
        targets_list.append(np.repeat(pat[np.newaxis, :], len(all_flips), axis=0))

    Q = np.vstack(queries_list)
    T = np.vstack(targets_list)

    for seed in range(5):
        res = recall(W, Q, HopfieldConfig(update="async", max_iter=100), rng=np.random.default_rng(seed))
        assert np.all(res.status == "converged")
        assert np.array_equal(res.states, T)


# --------------------------------------------------------------------------- recall & recall_rates


def test_recall_and_recall_rates_tables_and_summary(letters_run: Path):
    s = _summary(letters_run)
    recall_rows = _read_csv(letters_run / "recall.csv")
    rate_rows = _read_csv(letters_run / "recall_rates.csv")

    n_levels = len(s["noise_levels"])
    trials = int(s["trials"])
    assert n_levels == 8 and trials == 50
    assert len(recall_rows) == 2 * 2 * 4 * n_levels * trials == 6400
    assert len(rate_rows) == 2 * 2 * n_levels == 32

    assert s["selected_async_recovery"][0] == 1.0
    # Worst set: only N is a fixed point, so every query ends in N (recovery 0.25 = N's own share) up
    # to level 0.36. At 11 flips some queries fall into the inverted N attractor, so it may dip below.
    worst = s["worst_async_recovery"]
    assert worst[:-1] == [0.25] * (n_levels - 1)
    assert 0.2 <= worst[-1] <= 0.25
    assert s["example_n_async_cycles"] == 0
    assert s["example_n_sync_cycles"] > 0
    assert s["selected_max_level_full_recovery"] is not None and s["selected_max_level_full_recovery"] >= 0.08


# --------------------------------------------------------------------------- spurious states


def test_spurious_state(letters_run: Path):
    letters, patterns, _ = load_alphabet("datos/letras_5x5.json")
    sel_idx = [letters.index(ch) for ch in ("G", "R", "T", "V")]
    P_sel = patterns[sel_idx]
    W = hebbian_weights(P_sel)

    s = _summary(letters_run)
    assert s["spurious_found"] is True
    assert s["spurious_letter"] in ["G", "R", "T", "V"]
    assert s["spurious_n_flips"] == 10
    assert isinstance(s["spurious_trial"], int)
    assert len(s["spurious_query"]) == 5 and len(s["spurious_state"]) == 5
    assert len(s["spurious_overlaps"]) == 4
    assert s["spurious_nearest_letter"] in ["G", "R", "T", "V"]
    assert 1 <= int(s["spurious_hamming_to_nearest"]) <= 25
    assert len(s["stored_energies"]) == 4

    sp_vec = _glyph_to_pattern(s["spurious_state"])
    assert bool(is_fixed_point(W, sp_vec[np.newaxis, :])[0]) is True
    cls, idx = classify_states(sp_vec[np.newaxis, :], P_sel)
    assert str(cls[0]) == "spurious" and int(idx[0]) == -1

# --------------------------------------------------------------------------- capacity & energy


def test_capacity_and_energy_tables(letters_run: Path):
    s = _summary(letters_run)
    assert s["capacity_015"] == 3.75
    assert s["n_store_exceeds_015"] is True

    cap_rows = _read_csv(letters_run / "capacity.csv")
    assert len(cap_rows) == 12
    random_rows = [r for r in cap_rows if r["kind"] == "random"]
    assert [int(r["p"]) for r in random_rows] == list(range(1, 11))
    assert float(random_rows[0]["frac_stable"]) == 1.0
    kinds = {r["kind"] for r in cap_rows}
    assert kinds == {"random", "selected", "worst"}

    assert s["energy_never_increased"] is True
    energy_rows = _read_csv(letters_run / "energy.csv")
    assert len(energy_rows) >= 6400
    by_qid: dict[str, list[float]] = {}
    for r in energy_rows:
        qid = r["query_id"]
        if "_async_" in qid:
            by_qid.setdefault(qid, []).append(float(r["energy"]))
    assert by_qid
    for seq in by_qid.values():
        if len(seq) > 1:
            assert np.all(np.diff(seq) <= 1e-9)

    trace_rows = _read_csv(letters_run / "energy_trace.csv")
    assert len(trace_rows) > 0


# --------------------------------------------------------------------------- determinism & manual selection


def test_same_seed_gives_byte_identical_csvs(tmp_path: Path, write_config):
    cfg = write_config(base="configs/hopfield_letters.json", plots=False, **{"params.n_repeats": 2})
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert main(["run", str(cfg), "--out-root", str(out_a)]) == 0
    assert main(["run", str(cfg), "--out-root", str(out_b)]) == 0
    for name in ("recall.csv", "energy.csv", "capacity.csv", "recall_by_seed.csv", "recall_repetitions.csv", "summary.json"):
        assert (out_a / "hopfield_letters" / name).read_bytes() == (
            out_b / "hopfield_letters" / name
        ).read_bytes(), name


def test_manual_letter_selection(tmp_path: Path, write_config):
    cfg = write_config(
        base="configs/hopfield_letters.json",
        plots=False,
        **{"params.letters": ["A", "E", "I", "O"], "params.trials": 5, "params.capacity_subsets": 10, "params.n_repeats": 2},
    )
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    s = _summary(out / "hopfield_letters")
    assert s["selection"] == "manual"
    assert s["selected_letters"] == ["A", "E", "I", "O"]


# --------------------------------------------------------------------------- hopfield_eval helpers


def test_hopfield_eval_helpers():
    assert n_flips_for(0.2, 25) == 5
    letters, patterns, shape = load_alphabet("datos/letras_5x5.json")
    P = patterns[:4]
    W = hebbian_weights(P)
    q = flip_bits(P[0], 3, 10, np.random.default_rng(0))
    assert q.shape == (10, 25)
    assert np.all(np.sum(q != P[0], axis=1) == 3)
    ov = overlaps(P, P)
    assert ov.shape == (4, 4) and np.allclose(np.diag(ov), 1.0)
    near = nearest_pattern(P, P)
    assert near.tolist() == [0, 1, 2, 3]
    margins = stability_margins(W, P)
    assert margins.shape == (4,)
    ranked = rank_subsets(dot_matrix(patterns), 4)
    assert len(ranked) == 14950


# --------------------------------------------------------------------------- validation & boundaries


INVALID_CASES = [
    {"params.n_repeats": 1}, {"params.n_repeats": True},
    {"params.letters": ["A", "B", "C", "9"]},
    {"params.letters": ["A", "A", "B", "C"]},
    {"params.letters": ["A", "B", "C"]},
    {"params.updates": ["greedy"]},
    {"params.updates": []},
    {"params.updates": ["async", "async"]},
    {"params.max_iter": 0},
    {"params.max_iter": 1001},
    {"params.noise_levels": []},
    {"params.noise_levels": [-0.1, 0.2]},
    {"params.noise_levels": [0.2, 1.1]},
    {"params.noise_levels": [0.2, 0.2]},
    {"params.noise_levels": [0.3, 0.2]},
    {"params.example_level": 0.25},
    {"params.trials": 0},
    {"params.trials": 1001},
    {"params.spurious_noise": 0},
    {"params.spurious_noise": 1.5},
    {"params.capacity_max_p": 0},
    {"params.capacity_max_p": 27},
    {"params.capacity_subsets": 0},
    {"params.capacity_subsets": 5001},
    {"params.unknown": 1},
]


@pytest.mark.parametrize("overrides", INVALID_CASES, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()))
def test_invalid_hopfield_letters_config_exits_2_and_creates_no_run_dir(
    tmp_path: Path, write_config, capsys, overrides
):
    cfg = write_config(base="configs/hopfield_letters.json", **overrides)
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not (out / "hopfield_letters").exists()


def test_invalid_alphabet_files_exit_2_and_create_no_run_dir(tmp_path: Path, write_config, capsys):
    base_alpha = json.loads((PROJECT_ROOT / "datos" / "letras_5x5.json").read_text(encoding="utf-8"))

    # 1. A 4-row letter
    alpha_4row = dict(base_alpha)
    alpha_4row["A"] = alpha_4row["A"][:4]
    p_4row = tmp_path / "alpha_4row.json"
    p_4row.write_text(json.dumps(alpha_4row), encoding="utf-8")

    # 2. A character other than '#' and '.'
    alpha_badchar = dict(base_alpha)
    alpha_badchar["B"] = ["####.", "#...#", "##X#.", "#...#", "####."]
    p_badchar = tmp_path / "alpha_badchar.json"
    p_badchar.write_text(json.dumps(alpha_badchar), encoding="utf-8")

    # 3. Two identical letters
    alpha_dup = dict(base_alpha)
    alpha_dup["B"] = list(alpha_dup["A"])
    p_dup = tmp_path / "alpha_dup.json"
    p_dup.write_text(json.dumps(alpha_dup), encoding="utf-8")

    for idx, bad_path in enumerate((p_4row, p_badchar, p_dup)):
        cfg = write_config(base="configs/hopfield_letters.json", alphabet=str(bad_path))
        out = tmp_path / f"out_{idx}"
        assert main(["run", str(cfg), "--out-root", str(out)]) == 2
        assert "error:" in capsys.readouterr().err
        assert not (out / "hopfield_letters").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"params.max_iter": 1},
        {"params.max_iter": 1000},
        {"params.noise_levels": [0.0, 1.0], "params.example_level": 0.0},
        {"params.spurious_noise": 1.0},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_boundary_values_pass_validation(write_config, overrides):
    cfg = write_config(base="configs/hopfield_letters.json", **overrides)
    config = runner.validate_config(load_config(cfg))
    assert config["kind"] == "hopfield_letters"


def test_repeated_letter_metrics_report_sample_spread_between_seeds(letters_run):
    summary = _summary(letters_run)
    assert summary["n_repeats"] == 2 and summary["seeds"] == [42, 43]
    observations = _read_csv(letters_run / "recall_by_seed.csv")
    repeated = _read_csv(letters_run / "recall_repetitions.csv")
    for row in repeated:
        group = [r for r in observations if all(r[key] == row[key] for key in ("set", "update", "level"))]
        assert len(group) == 2
        assert all(int(r["n_queries"]) == 200 for r in group)
        values = np.array([float(r["recovery_rate"]) for r in group])
        assert float(row["recovery_rate_mean"]) == pytest.approx(values.mean())
        assert float(row["recovery_rate_std"]) == pytest.approx(values.std(ddof=1), abs=1e-8)
    capacity = _read_csv(letters_run / "capacity_by_seed.csv")
    assert len(capacity) == 2 * 10
