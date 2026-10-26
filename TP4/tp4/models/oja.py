"""Regla de Oja en línea, con η constante e inicio uniforme (clase 6, pp. 15–16)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MAX_W_NORM = 1e6


@dataclass(frozen=True)
class OjaConfig:
    epochs: int = 200
    eta0: float = 1e-3


@dataclass(frozen=True)
class OjaResult:
    config: OjaConfig
    weights: np.ndarray  # final raw w as trained; all NaN when diverged
    initial_weights: np.ndarray
    history: dict[str, np.ndarray]  # epoch, eta, w_norm, delta_w, cos_to_reference
    diverged: bool
    divergence_epoch: int | None


def _is_count(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _is_positive_real(v) -> bool:
    return (
        isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, (bool, np.bool_))
        and math.isfinite(v)
        and v > 0
    )


def _check_config(config: OjaConfig) -> None:
    if not (_is_count(config.epochs) and config.epochs >= 1):
        raise ValueError(f"epochs must be an integer >= 1, got {config.epochs!r}")
    if not _is_positive_real(config.eta0):
        raise ValueError(f"eta0 must be a finite number > 0, got {config.eta0!r}")


def _check_data(X, reference) -> tuple[np.ndarray, np.ndarray | None]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (samples x features), got {X.ndim}-D")
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError(f"X needs at least 2 rows and 1 column, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X holds non-finite values")
    if reference is None:
        return X, None
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (X.shape[1],):
        raise ValueError(f"reference must have shape ({X.shape[1]},), got {ref.shape}")
    if not np.isfinite(ref).all() or not float(np.linalg.norm(ref)) > 0.0:
        raise ValueError("reference must be finite and have nonzero norm")
    return X, ref


def train_oja(
    X: np.ndarray, config: OjaConfig, rng: np.random.Generator, reference: np.ndarray | None = None
) -> OjaResult:
    """Train one Oja neuron on the rows of X (already standardized); see the module docstring."""
    _check_config(config)
    X, ref = _check_data(X, reference)
    n, d = X.shape
    w = rng.uniform(0.0, 1.0, size=d)  # host RNG first: a fixed seed fixes the init and the orders
    w0 = w.copy()
    ref_norm = float(np.linalg.norm(ref)) if ref is not None else 1.0

    hist: dict[str, list[float]] = {"epoch": [], "eta": [], "w_norm": [], "delta_w": [], "cos_to_reference": []}
    diverged = False
    divergence_epoch: int | None = None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for epoch in range(config.epochs):
            w_prev = w
            eta_t = config.eta0
            for i in rng.permutation(n):
                x = X[int(i)]
                y = float(w @ x)
                w = w + eta_t * y * (x - y * w)
            norm = float(np.linalg.norm(w))
            if not bool(np.isfinite(w).all()) or not norm <= MAX_W_NORM:
                hist["epoch"].append(epoch + 1)
                hist["eta"].append(eta_t)
                hist["w_norm"].append(float("nan"))
                hist["delta_w"].append(float("nan"))
                hist["cos_to_reference"].append(float("nan"))
                diverged = True
                divergence_epoch = epoch + 1
                w = np.full(d, np.nan)
                break
            if ref is not None and norm > 0.0:
                cos = min(1.0, abs(float(w @ ref)) / (norm * ref_norm))
            else:
                cos = float("nan")
            hist["epoch"].append(epoch + 1)
            hist["eta"].append(eta_t)
            hist["w_norm"].append(norm)
            hist["delta_w"].append(float(np.linalg.norm(w - w_prev)))
            hist["cos_to_reference"].append(cos)
    return OjaResult(
        config=config,
        weights=w,
        initial_weights=w0,
        history={k: np.asarray(v, dtype=float) for k, v in hist.items()},
        diverged=diverged,
        divergence_epoch=divergence_epoch,
    )


def norm_band_epoch(norms, tol: float) -> int | None:
    """First 1-based epoch e with |norms[j] - 1| <= tol for every j >= e - 1 (band inclusive).

    NaN never counts as inside. None when the sequence is empty or its last value is outside.
    """
    values = np.asarray(norms, dtype=float)
    if values.size == 0:
        return None
    with np.errstate(invalid="ignore"):
        inside = np.abs(values - 1.0) <= tol  # NaN compares False, so it counts as outside
    if not bool(inside[-1]):
        return None
    outside = np.flatnonzero(~inside)
    return 1 if outside.size == 0 else int(outside[-1]) + 2
