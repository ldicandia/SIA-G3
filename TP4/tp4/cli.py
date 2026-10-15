"""Command-line entry point: `python -m tp4 run <config> [--out-root DIR]`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .errors import Tp4Error
from .runner import run_config_file
from .runs import DEFAULT_RUNS_ROOT, PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tp4", description="SIA TP4 experiment runner.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the experiment described by a JSON/TOML config")
    run.add_argument("config", type=Path, help="path to a .json or .toml experiment config")
    run.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="directory that receives runs/<experiment>/ (default: TP4/runs)",
    )
    return parser


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_dir, _summary = run_config_file(args.config, args.out_root)
    except Tp4Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"run dir: {_display(run_dir)}")
    print(f"summary: {_display(run_dir / 'summary.json')}")
    return 0
