"""The plotter turns run JSON into PNGs (ANL-01, ANL-02) and refuses an empty input dir."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def run_plotter(plot_script: Path, in_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(plot_script), "--in", str(in_dir), "--out", str(out_dir)],
        check=True,
        capture_output=True,
        text=True,
    )


def assert_png(path: Path) -> None:
    assert path.is_file(), f"missing {path}"
    assert path.stat().st_size > 0, f"empty {path}"


def test_plotter_writes_loss_png_for_and(run_validate, plot_script: Path, tmp_path: Path) -> None:
    result = run_validate("and")
    out_dir = tmp_path / "plots"

    run_plotter(plot_script, result.path.parent, out_dir)

    assert_png(out_dir / "and_loss.png")


def test_plotter_writes_no_fit_png_for_and(run_validate, plot_script: Path, tmp_path: Path) -> None:
    result = run_validate("and")
    out_dir = tmp_path / "plots_and"

    run_plotter(plot_script, result.path.parent, out_dir)

    assert sorted(p.name for p in out_dir.iterdir()) == ["and_loss.png"]
    assert not (out_dir / "and_fit.png").exists()


def test_plotter_writes_loss_and_fit_png_for_linear(run_validate, plot_script: Path, tmp_path: Path) -> None:
    result = run_validate("linear")
    out_dir = tmp_path / "plots_linear"

    run_plotter(plot_script, result.path.parent, out_dir)

    assert_png(out_dir / "linear_loss.png")
    assert_png(out_dir / "linear_fit.png")


def test_plotter_exits_1_on_empty_input_dir(plot_script: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    out_dir = tmp_path / "plots2"

    completed = subprocess.run(
        [sys.executable, str(plot_script), "--in", str(empty), "--out", str(out_dir)],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "no run JSON found" in completed.stderr
    assert not out_dir.exists()


def test_plotter_writes_loss_and_fit_png_for_tanh(run_validate, plot_script: Path, tmp_path: Path) -> None:
    result = run_validate("tanh")
    out_dir = tmp_path / "plots_tanh"

    run_plotter(plot_script, result.path.parent, out_dir)

    assert_png(out_dir / "tanh_loss.png")
    assert_png(out_dir / "tanh_fit.png")


def test_plotter_handles_three_runs_in_one_dir(run_validate, plot_script: Path, tmp_path: Path) -> None:
    all_dir = tmp_path / "all"
    all_dir.mkdir()
    for case in ("and", "linear", "tanh"):
        result = run_validate(case, extra=())
        shutil.copy(result.path, all_dir / result.path.name)
    out_dir = tmp_path / "plots_all"

    run_plotter(plot_script, all_dir, out_dir)

    expected = ["and_loss.png", "linear_fit.png", "linear_loss.png", "tanh_fit.png", "tanh_loss.png"]
    assert sorted(p.name for p in out_dir.iterdir()) == expected
    for name in expected:
        assert_png(out_dir / name)


def test_plotter_writes_xor_loss_png_for_xor(run_validate, plot_script: Path, tmp_path: Path) -> None:
    result = run_validate("xor")
    out_dir = tmp_path / "plots_xor"

    run_plotter(plot_script, result.path.parent, out_dir)

    assert_png(out_dir / "xor_loss.png")
    assert_png(out_dir / "xor_221_loss.png")
    assert_png(out_dir / "xor_2321_loss.png")
    assert_png(out_dir / "xor_step_loss.png")
