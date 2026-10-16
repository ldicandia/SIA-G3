"""CPU-parallel map over independent work items (spawn process pool).

Used by the `sweep` experiment kind; Phase 2 (Kohonen hyperparameter grid) and
Phase 4 (Hopfield noise sweeps) reuse it. Results come back in submission
order so downstream CSVs are deterministic regardless of the worker count.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

import numpy as np

T = TypeVar("T")
R = TypeVar("R")


def default_workers() -> int:
    """One worker per CPU, leaving one core free (never below 1)."""
    return max(1, (os.cpu_count() or 1) - 1)


def spawn_seeds(seed: int, n: int) -> list[int]:
    """`n` independent, deterministic child seeds derived from `seed`."""
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n)]


def run_parallel(fn: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    """`[fn(x) for x in items]`, optionally across worker processes, order preserved.

    `fn` and the items must be picklable (a module-level function and plain
    data). The pool uses the "spawn" start method so workers never inherit
    forked BLAS or matplotlib state. The first child exception is re-raised.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers!r}")
    if not items:
        return []
    if workers == 1 or len(items) == 1:
        return [fn(x) for x in items]
    max_workers = min(workers, len(items), os.cpu_count() or 1)
    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        return list(executor.map(fn, items))
