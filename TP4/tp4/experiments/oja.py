"""Experiment kind `oja`: the hand-written Oja neuron as a creator index (Ej. 1.3).

D-01: the input is the PCA input. `load_design` selects the 25 features of the feature set and
`standardize` z-scores them, exactly as in the pca and som kinds.
D-05: the trained vector is normalized to unit length and sign-oriented like
`tp4.analysis.pca.orient_components` (largest |coefficient| positive); the index of a creator is
its projection `z . w_hat`, on the same scale as the PC1 scores.
PC1 is recomputed with `fit_pca` on the same Z (identical to runs/pca) and is used only to
report the validation numbers (cosine, Pearson, per-epoch cosine, weight comparison); it never
touches the training.

Artifacts:
  history.csv   per epoch: eta, ||w||, ||delta w|| and |cos(w, PC1)|
  ranking.csv   one row per creator, best index first (equal indices keep the CSV order)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.pca import fit_pca, orient_components, standardize
from ..errors import ConfigError
from ..features import dominant_topic, load_design
from ..models.oja import OjaConfig, OjaResult, convergence_epoch, train_oja
from ..models.som import DECAYS
from ..runs import Experiment, RunContext, write_csv

EXTRA_KEYS = frozenset({"data", "feature_set"})
PARAM_DEFAULTS: dict[str, Any] = {
    "epochs": 200,
    "eta0": 0.001,
    "eta_min": 1e-05,
    "eta_decay": "constant",
    "norm_tol": 0.01,
}
MAX_EPOCHS = 5000


def _is_int(v: Any) -> bool:
    return type(v) is int


def _is_real(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def validate(config: dict) -> None:
    for key in ("data", "feature_set"):
        v = config.get(key)
        if not isinstance(v, str) or not v:
            raise ConfigError(f"oja: {key} must be a non-empty string path, got {v!r}")
    p = config["params"]
    if not (_is_int(p["epochs"]) and 1 <= p["epochs"] <= MAX_EPOCHS):
        raise ConfigError(f"oja: params.epochs must be an integer in [1, {MAX_EPOCHS}], got {p['epochs']!r}")
    eta0, eta_min = p["eta0"], p["eta_min"]
    if not (_is_real(eta0) and _is_real(eta_min) and 0 < eta_min <= eta0 <= 1):
        raise ConfigError(f"oja: need 0 < params.eta_min <= params.eta0 <= 1, got eta_min={eta_min!r}, eta0={eta0!r}")
    if p["eta_decay"] not in DECAYS:
        raise ConfigError(f"oja: params.eta_decay must be one of {list(DECAYS)}, got {p['eta_decay']!r}")
    tol = p["norm_tol"]
    if not (_is_real(tol) and 0 < tol < 1):
        raise ConfigError(f"oja: params.norm_tol must be a number with 0 < norm_tol < 1, got {tol!r}")


def oja_config_from_params(params) -> OjaConfig:
    return OjaConfig(
        epochs=int(params["epochs"]),
        eta0=float(params["eta0"]),
        eta_min=float(params["eta_min"]),
        eta_decay=params.get("eta_decay", "constant"),
    )


@dataclass(frozen=True)
class IndexFit:
    result: OjaResult
    w_hat: np.ndarray | None  # unit length, oriented like PC1 (D-05); None when the run diverged
    index: np.ndarray | None  # Z @ w_hat
    flipped: bool | None  # True when the orientation changed the sign of the trained vector


def fit_index(Z: np.ndarray, cfg: OjaConfig, rng: np.random.Generator, reference: np.ndarray | None) -> IndexFit:
    """Train the neuron and turn its vector into the oriented creator index (shared with the campaigns kind)."""
    result = train_oja(Z, cfg, rng, reference=reference)
    if result.diverged:
        return IndexFit(result=result, w_hat=None, index=None, flipped=None)
    u = result.weights / np.linalg.norm(result.weights)
    oriented, _ = orient_components(u[None, :], (Z @ u)[:, None])
    w_hat = oriented[0]
    return IndexFit(result=result, w_hat=w_hat, index=Z @ w_hat, flipped=bool(w_hat @ u < 0))


def rank_order(values) -> np.ndarray:
    """Creator positions from the highest value down; equal values keep their CSV order."""
    return np.argsort(-np.asarray(values, dtype=float), kind="stable")


def _top_feature(coefs: np.ndarray, features: list[str]) -> str:
    return features[int(np.argmax(np.abs(coefs)))]  # first maximum, as orient_components


def run(ctx: RunContext) -> dict:
    params = ctx.params
    cfg = oja_config_from_params(params)
    design = load_design(ctx.config["data"], ctx.config["feature_set"])
    Z = standardize(design.X)[0]
    features = list(design.X.columns)
    pca = fit_pca(Z, features)  # same oriented PC1 as runs/pca (full SVD, deterministic)
    pc1 = pca.components[0]

    # Training below is the only consumer of ctx.rng; every reading afterwards is deterministic.
    fit = fit_index(Z, cfg, ctx.rng, pc1)
    result = fit.result

    history = pd.DataFrame(result.history).rename(columns={"cos_to_reference": "cos_to_pc1"})
    write_csv(ctx.artifact("history.csv"), history)

    summary: dict[str, Any] = {
        "input": "features",
        "n_samples": int(Z.shape[0]),
        "n_features": int(Z.shape[1]),
        "epochs": cfg.epochs,
        "eta0": cfg.eta0,
        "eta_min": cfg.eta_min,
        "eta_decay": cfg.eta_decay,
        "norm_tol": float(params["norm_tol"]),
        "init": "uniform01",
        "diverged": bool(result.diverged),
        "divergence_epoch": result.divergence_epoch,
        "backend_resolved": ctx.backend.name,
    }
    if result.diverged:
        summary.update(
            {
                "final_w_norm": None,
                "epochs_to_converge": None,
                "cosine_to_pc1": None,
                "pearson_scores_pc1": None,
                "top_creator": None,
                "bottom_creator": None,
            }
        )
        return summary

    index, w_hat = fit.index, fit.w_hat
    order = rank_order(index)
    topics = dominant_topic(design.raw)
    ids = design.ids.reset_index(drop=True)
    ranking = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "creator_id": ids["creator_id"].to_numpy()[order],
            "name": ids["name"].to_numpy()[order],
            "dominant_topic": topics.to_numpy()[order],
            "followers": design.raw["followers"].to_numpy()[order],
            "oja_index": index[order],
            "pc1_score": pca.scores[order, 0],
        }
    )
    write_csv(ctx.artifact("ranking.csv"), ranking)

    norms = result.history["w_norm"]
    names = ranking["name"].tolist()
    summary.update(
        {
            "final_w_norm": round(float(norms[-1]), 4),
            "epochs_to_converge": convergence_epoch(norms, float(params["norm_tol"])),
            "cosine_to_pc1": round(min(1.0, abs(float(w_hat @ pc1))), 6),
            "pearson_scores_pc1": round(float(np.corrcoef(index, pca.scores[:, 0])[0, 1]), 6),
            "top_creator": names[0],
            "bottom_creator": names[-1],
        }
    )
    return summary


EXPERIMENT = Experiment(
    kind="oja",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
