"""Unit tests for scripts/audit_claims.py's claim-to-column audit.

All unit-level tests use small in-memory strings and tmp_path-based synthetic
fixtures (a fake plots/ directory plus a small hand-built dict standing in
for FIGURE_CLAIMS) rather than the real scripts.generate_plots.FIGURE_CLAIMS,
so they never require a real 75-run matrix to run. The one subprocess-level
test at the bottom exercises the real deck and real FIGURE_CLAIMS end to end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.audit_claims import audit, extract_referenced_figures, main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_extract_referenced_figures_returns_plots_only_filenames_only():
    markdown = (
        "# Slide\n"
        "![caption one](../plots/fig_a.png)\n"
        "some text\n"
        "![caption two](../plots/fig_a.png)\n"
        "![not a plot](../assets/flag.png)\n"
    )
    assert extract_referenced_figures(markdown) == {"fig_a.png"}


def test_extract_referenced_figures_on_no_image_links_returns_empty_set():
    markdown = "# Slide\nJust prose, no images at all.\n"
    assert extract_referenced_figures(markdown) == set()


def _write_markdown(path: Path, figures: list[str]) -> None:
    body = "# Deck\n" + "\n".join(f"![caption]({fig})" for fig in figures) + "\n"
    path.write_text(body, encoding="utf-8")


def test_audit_flags_a_referenced_figure_not_in_figure_claims(tmp_path):
    markdown_path = tmp_path / "deck.md"
    _write_markdown(markdown_path, ["../plots/fig_unclaimed.png"])
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "fig_unclaimed.png").write_bytes(b"")

    problems = audit(markdown_path, plots_dir, figure_claims={})

    assert len(problems) >= 1
    assert any("fig_unclaimed.png" in p for p in problems)


def test_audit_flags_a_claimed_figure_missing_from_disk(tmp_path):
    markdown_path = tmp_path / "deck.md"
    _write_markdown(markdown_path, ["../plots/fig_a.png"])
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    # Deliberately do NOT create fig_a.png on disk.

    problems = audit(markdown_path, plots_dir, figure_claims={"fig_a.png": "claim"})

    assert len(problems) >= 1
    assert any("fig_a.png" in p for p in problems)


def test_audit_flags_a_figure_claims_key_never_referenced(tmp_path):
    markdown_path = tmp_path / "deck.md"
    _write_markdown(markdown_path, ["../plots/fig_a.png"])
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "fig_a.png").write_bytes(b"")
    (plots_dir / "fig_unused.png").write_bytes(b"")

    problems = audit(
        markdown_path,
        plots_dir,
        figure_claims={"fig_a.png": "claim a", "fig_unused.png": "claim unused"},
    )

    assert len(problems) >= 1
    assert any("fig_unused.png" in p for p in problems)


def test_audit_on_a_fully_consistent_fixture_returns_empty_list(tmp_path):
    markdown_path = tmp_path / "deck.md"
    _write_markdown(markdown_path, ["../plots/fig_a.png", "../plots/fig_b.png"])
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "fig_a.png").write_bytes(b"")
    (plots_dir / "fig_b.png").write_bytes(b"")

    problems = audit(
        markdown_path,
        plots_dir,
        figure_claims={"fig_a.png": "claim a", "fig_b.png": "claim b"},
    )

    assert problems == []


def test_main_exits_0_against_the_real_deck_and_real_plots(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/audit_claims.py", "--markdown", "docs/presentacion.md", "--plots-dir", "plots"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_main_exits_nonzero_and_names_the_offending_file_on_a_corrupted_deck(tmp_path):
    real_deck = (PROJECT_ROOT / "docs" / "presentacion.md").read_text(encoding="utf-8")
    corrupted = real_deck.replace("fig_crossover_control.png", "fig_does_not_exist.png")
    corrupted_path = tmp_path / "corrupted_deck.md"
    corrupted_path.write_text(corrupted, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/audit_claims.py", "--markdown", str(corrupted_path), "--plots-dir", "plots"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "fig_does_not_exist.png" in result.stderr


def test_main_function_returns_int_directly(tmp_path):
    markdown_path = tmp_path / "deck.md"
    _write_markdown(markdown_path, ["../plots/fig_a.png"])
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "fig_a.png").write_bytes(b"")

    # main() with an unclaimed figure (no --figure-claims override exists,
    # so this exercises the real FIGURE_CLAIMS import path against a
    # synthetic markdown/plots pair -- expect non-zero since fig_a.png is
    # not a real FIGURE_CLAIMS key).
    exit_code = main(["--markdown", str(markdown_path), "--plots-dir", str(plots_dir)])
    assert exit_code != 0
