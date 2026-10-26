"""Run directories: staged swap into runs/<experiment>/, plus JSON/CSV writers.

A run is first written to `runs/.staging-<experiment>-<pid>/` and swapped into
`runs/<experiment>/` only on success. An existing `runs/<experiment>/` is
replaced only if it holds a tp4 `run.json` marker; anything else is refused.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import EXPERIMENT_NAME_RE
from .errors import RunDirError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
RUN_MARKER = "run.json"


def resolve_project_path(p: str | Path) -> Path:
    """Absolute paths stay absolute; relative paths resolve against the TP4 root."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class RunContext:
    experiment: str
    kind: str
    seed: int
    rng: np.random.Generator
    config: Mapping[str, Any]
    params: Mapping[str, Any]
    run_dir: Path
    plots: bool

    def artifact(self, name: str) -> Path:
        return self.run_dir / name


@dataclass(frozen=True)
class Experiment:
    kind: str
    run: Callable[[RunContext], dict]
    extra_keys: frozenset[str]
    param_defaults: Mapping[str, Any]
    validate: Callable[[dict], None] | None = None
    # Optional per-kind hook for sweeps over this kind: called with the sweep's context and the
    # sweep.csv table after it is written; the returned dict is merged into the sweep summary.
    sweep_report: Callable[[RunContext, pd.DataFrame], dict] | None = None


@contextlib.contextmanager
def staged_run_dir(out_root: Path, experiment: str) -> Iterator[Path]:
    """Yield a staging dir; on success swap it in as `out_root/experiment`."""
    if not isinstance(experiment, str) or not EXPERIMENT_NAME_RE.fullmatch(experiment):
        raise RunDirError(f"invalid experiment name {experiment!r}")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    root = out_root.resolve()
    final = out_root / experiment
    if final.resolve().parent != root:
        raise RunDirError(f"run dir for {experiment!r} resolves outside {out_root}")
    if final.exists() or final.is_symlink():
        if final.is_symlink() or not final.is_dir() or not (final / RUN_MARKER).is_file():
            raise RunDirError(
                f"refusing to replace {final}: it exists but holds no {RUN_MARKER} "
                "(not a tp4 run directory); move or delete it first"
            )

    staging = out_root / f".staging-{experiment}-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        yield staging
        if final.exists():
            if not (final / RUN_MARKER).is_file():
                raise RunDirError(f"refusing to replace {final}: it holds no {RUN_MARKER}")
            shutil.rmtree(final)
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and NaN/inf to JSON-safe values."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    text = json.dumps(_jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
