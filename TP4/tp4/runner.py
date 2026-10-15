"""Config-to-run-dir pipeline shared by every experiment kind."""

from __future__ import annotations

import platform
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .backend import resolve_backend
from .config import COMMON_KEYS, load_config, merge_params, validate_common
from .errors import ConfigError
from .experiments import get_experiment
from .runs import (
    DEFAULT_RUNS_ROOT,
    PROJECT_ROOT,
    RUN_MARKER,
    RunContext,
    staged_run_dir,
    write_json,
)

_VERSION_PACKAGES = ("numpy", "pandas", "matplotlib", "seaborn", "scikit-learn")
_SCALARS = (str, int, float, bool, type(None))


def validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a raw config; returns a new dict with defaults and merged params."""
    config = validate_common(raw)
    experiment = get_experiment(config["kind"])
    allowed = set(COMMON_KEYS) | set(experiment.extra_keys) | {"params"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ConfigError(f"unknown config key(s) for kind {config['kind']!r}: {unknown}")
    config["params"] = merge_params(config.get("params"), experiment.param_defaults, kind=config["kind"])
    if experiment.validate is not None:
        experiment.validate(config)
    return config


def _clean_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise ConfigError("experiment summary must be a dict")
    out: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, list):
            value = [v.item() if isinstance(v, np.generic) else v for v in value]
            ok = all(isinstance(v, _SCALARS) for v in value)
        else:
            ok = isinstance(value, _SCALARS)
        if not isinstance(key, str) or not ok:
            raise ConfigError(f"summary entry {key!r} must be a scalar or a list of scalars")
        out[key] = value
    return out


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in _VERSION_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return proc.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _portable_config_path(config_path: Path | None) -> str | None:
    if config_path is None:
        return None
    resolved = Path(config_path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def execute(
    config: Mapping[str, Any], run_dir: Path, *, config_path: Path | None = None
) -> dict[str, Any]:
    """Run the experiment for an already-validated config inside `run_dir`."""
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    experiment = get_experiment(config["kind"])
    backend = resolve_backend(config["backend"])
    ctx = RunContext(
        experiment=config["experiment"],
        kind=config["kind"],
        seed=config["seed"],
        rng=np.random.default_rng(config["seed"]),
        config=config,
        params=config["params"],
        run_dir=Path(run_dir),
        plots=config["plots"],
        backend=backend,
    )
    summary = _clean_summary(experiment.run(ctx))
    write_json(ctx.artifact("summary.json"), summary)

    finished = datetime.now(UTC)
    artifacts = sorted(p.name for p in Path(run_dir).iterdir() if p.is_file() and p.name != RUN_MARKER)
    write_json(
        ctx.artifact(RUN_MARKER),
        {
            "experiment": config["experiment"],
            "kind": config["kind"],
            "seed": config["seed"],
            "backend": backend.as_record(),
            "config": dict(config),
            "config_path": _portable_config_path(config_path),
            "versions": _versions(),
            "git_sha": _git_sha(),
            "artifacts": artifacts,
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        },
    )
    return summary


def run_config_file(
    config_path: str | Path, out_root: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Load, validate and run a config file; every validation error fires before runs/ is touched."""
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    config = validate_config(load_config(path))
    root = Path(out_root) if out_root is not None else DEFAULT_RUNS_ROOT
    with staged_run_dir(root, config["experiment"]) as staging:
        summary = execute(config, staging, config_path=path)
    return root / config["experiment"], summary
