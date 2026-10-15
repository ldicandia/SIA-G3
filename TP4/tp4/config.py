"""Config loading and validation (JSON primary, TOML by `.toml` extension).

Only `json.load` and `tomllib.load` are used: no eval, pickle, yaml or dynamic
imports driven by config values.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import ConfigError

EXPERIMENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
COMMON_KEYS = ("experiment", "kind", "seed", "backend", "plots")
SUPPORTED_BACKENDS = ("numpy", "cupy")


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON or TOML config file into a dict."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in (".json", ".toml"):
        raise ConfigError(f"unsupported config format {p.name!r}: use a .json or .toml file")
    try:
        if suffix == ".json":
            with p.open(encoding="utf-8") as handle:
                result = json.load(handle)
        else:
            with p.open("rb") as handle:
                result = tomllib.load(handle)
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"invalid config {p}: {exc}") from exc
    if not isinstance(result, dict):
        raise ConfigError(f"invalid config {p}: root must be an object/table")
    return result


def validate_common(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the keys every experiment shares; returns a NEW dict with defaults."""
    out = dict(raw)

    experiment = out.get("experiment")
    if not isinstance(experiment, str) or not EXPERIMENT_NAME_RE.fullmatch(experiment):
        raise ConfigError(
            f"experiment must match {EXPERIMENT_NAME_RE.pattern} "
            f"(lowercase letters, digits, '_' or '-'), got {experiment!r}"
        )

    kind = out.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ConfigError(f"kind must be a non-empty string, got {kind!r}")

    if "seed" not in out:
        raise ConfigError("seed is required (an integer >= 0)")
    seed = out["seed"]
    if type(seed) is not int or seed < 0:
        raise ConfigError(f"seed must be an integer >= 0, got {seed!r}")

    backend = out.get("backend", "numpy")
    if backend not in SUPPORTED_BACKENDS:
        raise ConfigError(f"backend must be one of {list(SUPPORTED_BACKENDS)}, got {backend!r}")
    out["backend"] = backend

    plots = out.get("plots", True)
    if type(plots) is not bool:
        raise ConfigError(f"plots must be a boolean, got {plots!r}")
    out["plots"] = plots
    return out


def _type_ok(value: Any, default: Any) -> bool:
    if default is None:
        return True
    if isinstance(default, bool):
        return type(value) is bool
    if isinstance(default, int):
        return type(value) is int
    if isinstance(default, float):
        return type(value) in (int, float)
    if isinstance(default, list):
        return isinstance(value, list)
    if isinstance(default, str):
        return isinstance(value, str)
    if isinstance(default, dict):
        return isinstance(value, dict)
    return isinstance(value, type(default))


def merge_params(params: Any, defaults: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    """Overlay `params` on `defaults`, rejecting unknown keys and wrong types."""
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ConfigError(f"params for kind {kind!r} must be an object/table, got {type(params).__name__}")
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        raise ConfigError(
            f"unknown param(s) for kind {kind!r}: {unknown}; allowed: {sorted(defaults)}"
        )
    merged = dict(defaults)
    for key, value in params.items():
        if not _type_ok(value, defaults[key]):
            raise ConfigError(
                f"param {key!r} for kind {kind!r} must be {type(defaults[key]).__name__}, got {value!r}"
            )
        merged[key] = value
    return merged
