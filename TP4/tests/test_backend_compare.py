"""The numpy-vs-cupy comparison tool, exercised without CUDA (a numpy-delegating fake cupy)."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from tp4.analysis import backend_compare as bc


def _fake_cupy(devices: int = 1) -> types.ModuleType:
    """A stand-in `cupy` that delegates every array function to numpy."""
    fake = types.ModuleType("cupy")
    fake.__getattr__ = lambda name: getattr(np, name)  # PEP 562 module __getattr__
    fake.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: devices))
    return fake


SMALL_LETTERS = {"params.trials": 5, "params.capacity_subsets": 4}


def test_main_without_cupy_reports_unavailable_runs_nothing_and_returns_1(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.warns(RuntimeWarning, match="cupy"):
        code = bc.main([])
    assert code == 1
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("backend unavailable:")]
    assert len(lines) == 1 and "cupy" in lines[0]
    assert "backend_compare" not in out  # no table, no verdict: nothing ran


def test_main_with_fake_cupy_reports_equivalence_and_writes_json(
    monkeypatch, capsys, write_config, tmp_path
):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    cfg = write_config("configs/hopfield_letters.json", **SMALL_LETTERS)
    report_path = tmp_path / "report.json"
    code = bc.main(["--configs", str(cfg), "--json", str(report_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "backend_compare ok: 1 configs equivalent" in out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    (entry,) = report["configs"]
    assert entry["equivalent"] is True and entry["max_abs_diff"] == 0
    assert entry["numpy_seconds"] > 0 and entry["cupy_seconds"] > 0
    assert entry["speedup"] > 0
    assert {p["name"] for p in report["probes"]} and all(p["ok"] for p in report["probes"])


def test_main_default_out_root_leaves_no_runs_behind(monkeypatch, write_config, project_root):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    before = sorted(p.name for p in (project_root / "runs").glob("*")) if (project_root / "runs").exists() else []
    cfg = write_config("configs/hopfield_letters.json", **SMALL_LETTERS)
    assert bc.main(["--configs", str(cfg)]) == 0
    after = sorted(p.name for p in (project_root / "runs").glob("*")) if (project_root / "runs").exists() else []
    assert before == after


def test_sweep_config_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    code = bc.main(["--configs", "configs/oja_sweep.json"])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_missing_config_is_a_usage_error(monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    assert bc.main(["--configs", str(tmp_path / "nope.json")]) == 2
    assert "error:" in capsys.readouterr().err


# --------------------------------------------------------------------------- artifact comparison


def _write_run(path, summary, csv_text):
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "table.csv").write_text(csv_text, encoding="utf-8")
    return path


BASE_SUMMARY = {"backend_resolved": "numpy", "score": 1.5, "n": 3, "label": "a", "ok": True, "items": [1.0, 2.0]}
BASE_CSV = "id,label,x,count\n1,a,0.5,3\n2,b,0.25,4\n"


def _pair(tmp_path, summary_b=None, csv_b=None):
    a = _write_run(tmp_path / "a", BASE_SUMMARY, BASE_CSV)
    summary = dict(BASE_SUMMARY, backend_resolved="cupy") if summary_b is None else summary_b
    b = _write_run(tmp_path / "b", summary, BASE_CSV if csv_b is None else csv_b)
    return a, b


def test_identical_runs_are_equivalent_and_backend_resolved_is_ignored(tmp_path):
    a, b = _pair(tmp_path)
    max_diff, mismatches = bc.compare_run_dirs(a, b, rtol=1e-6, atol=1e-9)
    assert mismatches == [] and max_diff == 0


def test_difference_inside_tolerance_is_accepted(tmp_path):
    a, b = _pair(tmp_path, csv_b="id,label,x,count\n1,a,0.5000000001,3\n2,b,0.25,4\n")
    max_diff, mismatches = bc.compare_run_dirs(a, b, rtol=1e-6, atol=1e-9)
    assert mismatches == []
    assert 0 < max_diff < 1e-9


def test_tampered_csv_cell_is_flagged(tmp_path):
    a, b = _pair(tmp_path, csv_b="id,label,x,count\n1,a,0.51,3\n2,b,0.25,4\n")
    max_diff, mismatches = bc.compare_run_dirs(a, b, rtol=1e-6, atol=1e-9)
    assert len(mismatches) == 1 and "table.csv" in mismatches[0]
    assert max_diff == pytest.approx(0.01)


def test_discrete_cells_that_differ_by_one_are_flagged(tmp_path):
    a, b = _pair(tmp_path, csv_b="id,label,x,count\n1,a,0.5,4\n2,b,0.25,4\n")
    _, mismatches = bc.compare_run_dirs(a, b, rtol=0.5, atol=0.5)  # even with a loose tolerance
    assert len(mismatches) == 1 and "count" in mismatches[0]
    a2, b2 = _pair(tmp_path / "labels", csv_b="id,label,x,count\n1,c,0.5,3\n2,b,0.25,4\n")
    _, mismatches = bc.compare_run_dirs(a2, b2, rtol=0.5, atol=0.5)
    assert len(mismatches) == 1 and "label" in mismatches[0]


def test_missing_artifact_and_row_count_are_flagged(tmp_path):
    a, b = _pair(tmp_path)
    (b / "table.csv").unlink()
    _, mismatches = bc.compare_run_dirs(a, b, rtol=1e-6, atol=1e-9)
    assert any("table.csv" in m and "missing" in m for m in mismatches)
    a2, b2 = _pair(tmp_path / "rows", csv_b="id,label,x,count\n1,a,0.5,3\n")
    _, mismatches = bc.compare_run_dirs(a2, b2, rtol=1e-6, atol=1e-9)
    assert any("rows" in m for m in mismatches)


def test_differing_summary_value_is_flagged(tmp_path):
    other = dict(BASE_SUMMARY, backend_resolved="cupy", score=1.6)
    a, b = _pair(tmp_path, summary_b=other)
    max_diff, mismatches = bc.compare_run_dirs(a, b, rtol=1e-6, atol=1e-9)
    assert len(mismatches) == 1 and "summary.json" in mismatches[0] and "score" in mismatches[0]
    assert max_diff == pytest.approx(0.1)
    for key, value in (("n", 4), ("label", "z"), ("ok", False), ("items", [1.0, 2.5])):
        a2, b2 = _pair(tmp_path / key, summary_b=dict(BASE_SUMMARY, backend_resolved="cupy", **{key: value}))
        _, mismatches = bc.compare_run_dirs(a2, b2, rtol=1e-6, atol=1e-9)
        assert mismatches, key


# --------------------------------------------------------------------------- probes


def test_a_failing_probe_makes_main_return_3(monkeypatch, write_config, capsys):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    monkeypatch.setattr(
        bc, "_probe_argmin_ties", lambda xp, rtol, atol: {"name": "argmin_ties", "ok": False, "detail": "boom"}
    )
    cfg = write_config("configs/hopfield_letters.json", **SMALL_LETTERS)
    code = bc.main(["--configs", str(cfg)])
    out = capsys.readouterr().out
    assert code == 3
    assert "backend_compare FAILED" in out and "boom" in out


def test_real_probes_pass_on_the_numpy_delegating_fake():
    fake = _fake_cupy()
    for probe in (bc._probe_argmin_ties, bc._probe_som_duplicate_prototypes):
        result = probe(fake, 1e-6, 1e-9)
        assert result["ok"] is True, result
