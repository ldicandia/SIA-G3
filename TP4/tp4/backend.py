"""Selectable array backend: numpy (default) or cupy (optional, with recorded fallback).

CUDA must never block a run or a test: when cupy is not importable, no CUDA
device is visible, or the driver misbehaves, `resolve_backend("cupy")` warns
and returns the numpy backend, keeping the reason so `run.json` can record it.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

from .errors import ConfigError

_MAX_REASON = 200


@dataclass(frozen=True)
class Backend:
    requested: str
    name: str
    xp: ModuleType
    fallback_reason: str | None

    def as_record(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved": self.name,
            "fallback_reason": self.fallback_reason,
        }


def _numpy_backend(requested: str, reason: str | None = None) -> Backend:
    return Backend(requested=requested, name="numpy", xp=np, fallback_reason=reason)


def _fallback(reason: str) -> Backend:
    reason = reason[:_MAX_REASON]
    warnings.warn(f"{reason}; falling back to numpy", RuntimeWarning, stacklevel=3)
    return _numpy_backend("cupy", reason)


def resolve_backend(requested: str = "numpy") -> Backend:
    """Return the requested backend, or numpy plus a reason when cupy cannot be used."""
    if requested == "numpy":
        return _numpy_backend("numpy")
    if requested != "cupy":
        raise ConfigError(f"unknown backend {requested!r}; use 'numpy' or 'cupy'")
    try:
        cupy = importlib.import_module("cupy")
        devices = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:  # any CUDA stack failure must degrade to numpy, never raise
        return _fallback(f"cupy unavailable: {type(exc).__name__}: {exc}")
    if devices < 1:
        return _fallback("cupy unavailable: no CUDA device")
    return Backend(requested="cupy", name="cupy", xp=cupy, fallback_reason=None)


def to_numpy(a: Any) -> np.ndarray:
    """Bring a backend array back to a host numpy array (cupy arrays expose `.get()`)."""
    if not isinstance(a, np.ndarray) and callable(getattr(a, "get", None)):
        return np.asarray(a.get())
    return np.asarray(a)
