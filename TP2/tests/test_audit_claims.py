"""Unit tests for scripts/audit_claims.py's claim-to-column audit.

The unit-level tests use tmp_path-based synthetic fixtures (a fake plots/
directory plus a small hand-built dict standing in for FIGURE_CLAIMS) rather
than the real scripts.generate_plots.FIGURE_CLAIMS, so they never require a
real matrix run. The subprocess-level tests at the bottom exercise the real
FIGURE_CLAIMS end to end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.audit_claims import audit, main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_audit_flags_a_claimed_figure_missing_from_disk(tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    # Deliberately do NOT create fig_a.png on disk.

    problems = audit(plots_dir, figure_claims={"fig_a.png": "claim"})

    assert len(problems) == 1
    assert "fig_a.png" in problems[0]


def test_audit_reports_every_missing_figure_not_only_the_first(tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "fig_present.png").write_bytes(b"")

    problems = audit(
        plots_dir,
        figure_claims={
            "fig_present.png": "claim present",
            "fig_missing_a.png": "claim a",
            "fig_missing_b.png": "claim b",
        },
    )

    assert len(problems) == 2
    assert any("fig_missing_a.png" in p for p in problems)
    assert any("fig_missing_b.png" in p for p in problems)


def test_audit_on_a_fully_consistent_fixture_returns_empty_list(tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "fig_a.png").write_bytes(b"")
    (plots_dir / "fig_b.png").write_bytes(b"")

    problems = audit(plots_dir, figure_claims={"fig_a.png": "claim a", "fig_b.png": "claim b"})

    assert problems == []


def test_audit_ignores_extra_files_that_are_not_claimed(tmp_path):
    """plots/ holds hand-added images alongside the generated figures; an
    unclaimed file on disk is not a violation, only a claim without a file is.
    """
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "fig_a.png").write_bytes(b"")
    (plots_dir / "some_screenshot.png").write_bytes(b"")

    assert audit(plots_dir, figure_claims={"fig_a.png": "claim a"}) == []


def test_main_exits_0_against_the_real_plots():
    result = subprocess.run(
        [sys.executable, "scripts/audit_claims.py", "--plots-dir", "plots"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_main_exits_nonzero_and_names_the_offending_figures_when_plots_are_absent(tmp_path):
    """The real FIGURE_CLAIMS against an empty plots dir: every claim is
    missing, so main must fail and name what it could not find.
    """
    empty_plots = tmp_path / "plots"
    empty_plots.mkdir()

    result = subprocess.run(
        [sys.executable, "scripts/audit_claims.py", "--plots-dir", str(empty_plots)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing figure on disk" in result.stderr


def test_main_function_returns_int_directly(tmp_path):
    # main() with an empty plots dir exercises the real FIGURE_CLAIMS import
    # path -- expect non-zero since none of the real figures exist there.
    empty_plots = tmp_path / "plots"
    empty_plots.mkdir()

    assert main(["--plots-dir", str(empty_plots)]) != 0
