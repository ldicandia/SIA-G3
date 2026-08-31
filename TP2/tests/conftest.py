"""Shared fixtures for the TP2 contract suite.

Every generator here is constructed per test. Nothing in this file — or in any
test that uses it — calls a global seeding function: a global seed leaks across
tests and turns an ordering change into a spurious failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pytest

from tp2 import cli
from tp2.engine.genome import random_population

SEED = 20260829

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    """The TP2 directory — the root `prepare_run_dir` refuses to write outside of."""
    return PROJECT_ROOT


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh seeded generator, never the global one."""
    return np.random.default_rng(SEED)


@pytest.fixture
def small_size() -> tuple[int, int]:
    """Canvases stay small so the whole suite stays under the ten-second budget."""
    return (64, 64)


@pytest.fixture
def fixed_genes() -> np.ndarray:
    """A deterministic 12-triangle chromosome, built from its own generator.

    Drawn locally rather than from the `rng` fixture so the value does not depend
    on whether a test consumed `rng` first.
    """
    return random_population(np.random.default_rng(SEED), 1, 12)[0]


@pytest.fixture
def flat_target() -> np.ndarray:
    """A small uniform target; the engine takes arrays, never paths."""
    return np.full((8, 8, 3), 255.0, dtype=np.float32)


@pytest.fixture
def target_image() -> Path:
    """A shipped asset, addressed absolutely so the tests do not depend on cwd."""
    return PROJECT_ROOT / "assets" / "flag_ar.png"


@pytest.fixture
def run_slice0(target_image: Path) -> Callable[..., Path]:
    """Drive `cli.main()` in-process and return the run directory it wrote.

    In-process rather than via subprocess to keep the suite fast; the subprocess
    path is already covered by `tests/test_no_pygame.py`.

    `--allow-outside` is passed because pytest's `tmp_path` is deliberately
    outside the TP2 root — that guard is exercised on its own in
    `tests/test_artifacts.py`, not incidentally by every other test.
    """

    def _run(
        out: str | Path,
        *,
        seed: int | None = 7,
        triangles: int = 8,
        population: int = 4,
        canvas: int = 32,
        extra: Sequence[str] = (),
    ) -> Path:
        argv = [
            "--image", str(target_image),
            "--triangles", str(triangles),
            "--population", str(population),
            "--canvas", str(canvas),
            "--out", str(out),
            "--allow-outside",
        ]
        if seed is not None:
            # `is not None`, never truthiness: seed 0 is a real seed.
            argv += ["--seed", str(seed)]
        argv += list(extra)
        assert cli.main(argv) == 0
        return Path(out)

    return _run


@pytest.fixture
def read_json() -> Callable[[Path], Any]:
    def _read(path: Path) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    return _read
