"""Shared fixtures: locate the built `tp3` binary and run `tp3 validate`.

The Python side never imports the C++ core. It only runs `build/tp3` as a
subprocess and reads the JSON the binary writes.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable, NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class RunResult(NamedTuple):
    data: dict
    path: Path
    stdout: str


@pytest.fixture(scope="session")
def tp3_bin() -> Path:
    """Path to the `tp3` binary; `TP3_BIN` overrides the default `build/tp3`.

    A missing binary skips (never passes) so a forgotten build is visible.
    """
    path = Path(os.environ.get("TP3_BIN", REPO_ROOT / "build" / "tp3"))
    if not path.is_file():
        pytest.skip(f"{path} not found — build it with: cmake -S . -B build && cmake --build build")
    return path


@pytest.fixture
def run_validate(tp3_bin: Path, tmp_path: Path) -> Callable[..., RunResult]:
    """Return a callable that runs `tp3 validate <case>` into a fresh dir under tmp_path."""
    counter = {"n": 0}

    def _run(case: str, seed: int = 42, extra: tuple[str, ...] = ()) -> RunResult:
        counter["n"] += 1
        out_dir = tmp_path / f"{case}_{seed}_{counter['n']}"
        completed = subprocess.run(
            [str(tp3_bin), "validate", case, "--seed", str(seed), "--out", str(out_dir), *extra],
            check=True,
            capture_output=True,
            text=True,
        )
        path = out_dir / f"{case}.json"
        return RunResult(data=json.loads(path.read_text()), path=path, stdout=completed.stdout)

    return _run


@pytest.fixture(scope="session")
def plot_script() -> Path:
    return REPO_ROOT / "scripts" / "plot_validation.py"
