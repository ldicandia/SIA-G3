"""Oja's single neuron written by hand in numpy (no ML library, no eigen-solver).

The course rule (clases/6 - Regla de Oja y Sanger, pages 12-16): with y = w . x,

    w <- w + eta * y * (x - y * w)          (i.e. delta w = eta * (y x - y^2 w))

applied online, one creator at a time, on standardized (zero-mean) data. The initial w is uniform
in [0, 1), drawn from the seeded rng before the loop. Every epoch visits the creators in one
`rng.permutation` order (D-02). eta is constant by default (the course's 10^-3); an optional
decay goes through `tp4.models.som.decay` over all T = epochs * n iterations (D-03).

The model knows nothing about PCA: an optional `reference` vector is used only to record
|cos(w, reference)| per epoch, never to initialize, update or stop the training.

A run that blows up is flagged (`diverged`, `divergence_epoch`, NaN weights), not raised: a
non-finite weight or a norm above MAX_W_NORM at an epoch end stops the loop. numpy overflow
warnings are suppressed only inside the training loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .som import DECAYS, decay

MAX_W_NORM = 1e6


@dataclass(frozen=True)
class OjaConfig:
    epochs: int = 200
    eta0: float = 1e-3
    eta_min: float = 1e-5
    eta_decay: str = "constant"


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
    for name, value in (("eta0", config.eta0), ("eta_min", config.eta_min)):
        if not _is_positive_real(value):
            raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    if config.eta_min > config.eta0:
        raise ValueError(f"eta_min must be <= eta0, got eta_min={config.eta_min!r}, eta0={config.eta0!r}")
    if config.eta_decay not in DECAYS:
        raise ValueError(f"unknown eta_decay {config.eta_decay!r}; allowed: {list(DECAYS)}")


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
    total = config.epochs * n
    ref_norm = float(np.linalg.norm(ref)) if ref is not None else 1.0

    hist: dict[str, list[float]] = {"epoch": [], "eta": [], "w_norm": [], "delta_w": [], "cos_to_reference": []}
    diverged = False
    divergence_epoch: int | None = None
    t = 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for epoch in range(config.epochs):
            w_prev = w
            eta_t = config.eta0
            for i in rng.permutation(n):
                eta_t = decay(config.eta0, config.eta_min, t, total, config.eta_decay)
                x = X[int(i)]
                y = float(w @ x)
                w = w + eta_t * y * (x - y * w)
                t += 1
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


def convergence_epoch(norms, tol: float) -> int | None:
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
