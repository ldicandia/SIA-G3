from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys

import pytest

from tp4.cli import main
from tp4.runs import PROJECT_ROOT, RUN_MARKER

DATA = PROJECT_ROOT / "datos" / "creadores.csv"


def _snapshot(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def test_end_to_end_run_writes_artifacts(out_root, capsys):
    assert main(["run", "configs/eda.json", "--out-root", str(out_root)]) == 0
    run_dir = out_root / "eda"
    for name in ("run.json", "summary.json", "columns.csv"):
        assert (run_dir / name).stat().st_size > 0
    assert "run dir:" in capsys.readouterr().out


def test_run_json_records_provenance(out_root, tmp_path, write_config):
    main(["run", "configs/eda.json", "--out-root", str(out_root)])
    run = json.loads((out_root / "eda" / "run.json").read_text(encoding="utf-8"))
    assert run["seed"] == 42
    assert run["kind"] == "eda"
    assert run["experiment"] == "eda"
    assert run["config"]["data"] == "datos/creadores.csv"
    assert run["config_path"] == "configs/eda.json"
    assert "numpy" in run["versions"]
    assert "columns.csv" in run["artifacts"] and "run.json" not in run["artifacts"]

    other = tmp_path / "other"
    cfg = write_config(plots=False)
    main(["run", str(cfg), "--out-root", str(other)])
    outside = json.loads((other / "eda" / "run.json").read_text(encoding="utf-8"))
    assert outside["config_path"] == cfg.name


def test_summary_matches_csv_shape(out_root, csv_rows, write_config):
    main(["run", str(write_config(plots=False)), "--out-root", str(out_root)])
    summary = json.loads((out_root / "eda" / "summary.json").read_text(encoding="utf-8"))
    with DATA.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert summary["n_rows"] == csv_rows(DATA)
    assert summary["n_columns"] == len(header)


def test_module_entry_point_subprocess(out_root):
    proc = subprocess.run(
        [sys.executable, "-m", "tp4", "run", "configs/eda.json", "--out-root", str(out_root)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_root / "eda" / "summary.json").is_file()


def test_same_seed_is_byte_reproducible(tmp_path, write_config):
    cfg = str(write_config(plots=False))  # PNG rendering is slow on the Windows mount; CSV/JSON are the contract
    a, b = tmp_path / "a", tmp_path / "b"
    assert main(["run", cfg, "--out-root", str(a)]) == 0
    assert main(["run", cfg, "--out-root", str(b)]) == 0
    for name in (p.name for p in (a / "eda").iterdir() if p.suffix in (".json", ".csv")):
        if name == RUN_MARKER:
            continue
        assert (a / "eda" / name).read_bytes() == (b / "eda" / name).read_bytes(), name


def test_json_and_toml_configs_are_equivalent(tmp_path, write_config):
    cfg_json = write_config(fmt="json", plots=False)
    cfg_toml = write_config(fmt="toml", plots=False)
    a, b = tmp_path / "a", tmp_path / "b"
    assert main(["run", str(cfg_json), "--out-root", str(a)]) == 0
    assert main(["run", str(cfg_toml), "--out-root", str(b)]) == 0
    assert (a / "eda" / "summary.json").read_bytes() == (b / "eda" / "summary.json").read_bytes()


@pytest.mark.parametrize(
    "overrides",
    [
        {"bogus": 1},
        {"seed": "__REMOVE__"},
        {"seed": "42"},
        {"seed": True},
        {"seed": -1},
        {"experiment": "../escape"},
        {"kind": "nope"},
        {"params.nope": 1},
        {"params.outlier_iqr_k": 0},
        {"backend": "torch"},
        {"plots": "yes"},
        {"data": "datos/does_not_exist.csv"},
    ],
    ids=lambda o: ",".join(f"{k}={v!r}" for k, v in o.items()),
)
def test_invalid_config_exits_2_and_keeps_previous_run(out_root, write_config, capsys, overrides):
    assert main(["run", str(write_config(plots=False)), "--out-root", str(out_root)]) == 0
    before = _snapshot(out_root / "eda")
    capsys.readouterr()

    cfg = write_config(**{"plots": False, **overrides})
    assert main(["run", str(cfg), "--out-root", str(out_root)]) == 2
    assert "error:" in capsys.readouterr().err
    assert _snapshot(out_root / "eda") == before
    assert not list(out_root.glob(".staging-*"))


def test_unreadable_or_unsupported_config_exits_2(tmp_path, out_root, capsys):
    assert main(["run", str(tmp_path / "missing.json"), "--out-root", str(out_root)]) == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: 1", encoding="utf-8")
    assert main(["run", str(bad), "--out-root", str(out_root)]) == 2
    assert capsys.readouterr().err.count("error:") == 2


def test_foreign_run_dir_is_refused_with_exit_2(out_root, capsys):
    (out_root / "eda").mkdir(parents=True)
    (out_root / "eda" / "mine.txt").write_text("keep", encoding="utf-8")
    assert main(["run", "configs/eda.json", "--out-root", str(out_root)]) == 2
    assert "error:" in capsys.readouterr().err
    assert (out_root / "eda" / "mine.txt").read_text() == "keep"


def test_source_csv_is_not_modified(out_root, write_config):
    before = hashlib.sha256(DATA.read_bytes()).hexdigest()
    main(["run", str(write_config(plots=False)), "--out-root", str(out_root)])
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == before
