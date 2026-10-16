"""Experiment kind `oja`: the hand-written Oja neuron as a creator index (Ej. 1.3).

D-01: the input is the PCA input. `load_design` selects the 25 features of the feature set and
`standardize` z-scores them, exactly as in the pca and som kinds.
D-05: the trained vector is normalized to unit length and sign-oriented like
`tp4.analysis.pca.orient_components` (largest |coefficient| positive); the index of a creator is
its projection `z . w_hat`, on the same scale as the PC1 scores.
PC1 is recomputed with `fit_pca` on the same Z (identical to runs/pca) and is used only to
report the validation numbers (cosine, Pearson, per-epoch cosine, weight comparison); it never
touches the training.

Config keys (besides the common ones):
  data, feature_set   the PCA input
  som_config          path to the `som` config whose creator regions go into ranking.csv; it must
                      use the same data and feature_set (D-08)

Artifacts:
  history.csv          per epoch: eta, ||w||, ||delta w|| and |cos(w, PC1)|
  ranking.csv          one row per creator, best index first (equal indices keep the CSV order),
                       with topic shares (art reconstructed), followers, index, PC1 score, SOM region
  weights_vs_pc1.csv   the 25 Oja weights (oriented, and as trained) next to the PC1 loadings
  convergence.png, cos_to_pc1.png, weights_vs_pc1.png, scores_vs_pc1.png
A diverging run (large eta) is flagged in summary.json; it writes history.csv and the
convergence plot but no ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.pca import fit_pca, orient_components, standardize
from ..analysis.som_maps import reconstruct_topic_art
from ..errors import ConfigError
from ..features import dominant_topic, load_design
from ..models.oja import OjaConfig, OjaResult, convergence_epoch, train_oja
from ..models.som import DECAYS
from ..plotting import plt, save_figure
from ..runs import Experiment, RunContext, resolve_project_path, write_csv
from .oja_sweep_report import report
from .pca import TOPIC_COLORS, TOPIC_LABELS
from .som import load_som_config, som_evidence

EXTRA_KEYS = frozenset({"data", "feature_set", "som_config"})
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
    for key in ("data", "feature_set", "som_config"):
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
    som = load_som_config(config["som_config"], who="oja")
    for key in ("data", "feature_set"):
        if resolve_project_path(som[key]).resolve() != resolve_project_path(config[key]).resolve():
            raise ConfigError(
                f"oja: som_config uses a different {key} ({som[key]!r}) than this config ({config[key]!r}); "
                "the SOM regions must come from the same input"
            )


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


# --------------------------------------------------------------------------- plots


def _plot_convergence(history, norm_tol, converge_epoch, divergence_epoch, path: Path) -> None:
    epoch, norm, delta = history["epoch"], history["w_norm"], history["delta_w"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    top.plot(epoch, norm, color="#0072B2", label="norma de w")
    top.axhline(1.0, color="#444444", linewidth=1.0, label="norma 1")
    top.axhspan(1.0 - norm_tol, 1.0 + norm_tol, color="#009E73", alpha=0.18, label=f"banda ±{norm_tol:g}")
    if converge_epoch is not None:
        top.axvline(converge_epoch, color="#D55E00", linestyle="--", linewidth=1.2,
                    label=f"converge en la época {converge_epoch}")
    if divergence_epoch is not None:
        top.axvline(divergence_epoch, color="#D55E00", linestyle="--", linewidth=1.2,
                    label=f"diverge en la época {divergence_epoch}")
        top.set_ylim(0, 5)
    top.set_ylabel("‖w‖")
    top.legend(loc="upper right", fontsize=8)
    top.grid(alpha=0.3)
    top.set_title("Regla de Oja: la norma de w tiende a 1")
    positive = np.isfinite(delta) & (delta > 0)
    bottom.plot(epoch, np.where(positive, delta, np.nan), color="#CC79A7")
    if positive.any():
        bottom.set_yscale("log")
    bottom.set_ylabel("‖Δw‖ en la época")
    bottom.set_xlabel("época")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, path)


def _plot_cosine(history, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["epoch"], history["cos_to_reference"], color="#0072B2")
    ax.axhline(1.0, color="#444444", linewidth=1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("época")
    ax.set_ylabel("|cos(w, PC1)|")
    ax.set_title("Alineación de w con PC1 por época (invariante al signo)")
    ax.grid(alpha=0.3)
    save_figure(fig, path)


def _plot_weights(table: pd.DataFrame, path: Path) -> None:
    ordered = table.reindex(table["pc1_loading"].abs().sort_values(ascending=False, kind="stable").index)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(ordered) + 2.0))
    ax.barh(y - 0.2, ordered["oja_weight"], height=0.4, color="#0072B2", label="Oja (ŵ)")
    ax.barh(y + 0.2, ordered["pc1_loading"], height=0.4, color="#E69F00", label="PC1")
    ax.set_yticks(y, labels=list(ordered["feature"]), fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("coeficiente (vector unitario)")
    ax.set_title("Pesos de Oja frente a las cargas de PC1, por variable")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    save_figure(fig, path)


def _plot_scores(index, pc1_scores, names, topics, order, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for topic, color in TOPIC_COLORS.items():
        mask = topics == topic
        if mask.any():
            ax.scatter(pc1_scores[mask], index[mask], s=42, color=color, edgecolor="white", linewidth=0.5,
                       label=TOPIC_LABELS[topic], zorder=3)
    lim = max(float(np.abs(pc1_scores).max()), float(np.abs(index).max())) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#444444", linewidth=1.0, linestyle="--", label="identidad")
    # Labels sit in two fixed columns over the empty corners (the data hug the diagonal) and join
    # their point with a thin leader line, so they cannot collide however close the points are.
    for group, x_label, ha, y_start, y_step in ((order[:5][::-1], 0.78, "left", 0.66, -0.05), (order[-5:][::-1], 0.22, "right", 0.54, -0.05)):
        for j, i in enumerate(group):
            ax.annotate(str(names[i]), (pc1_scores[i], index[i]), xytext=(x_label, y_start + y_step * j),
                        textcoords="axes fraction", ha=ha, va="center", fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": "#888888", "linewidth": 0.6, "shrinkA": 2, "shrinkB": 3})
    ax.set_xlabel("puntaje en PC1")
    ax.set_ylabel("índice de Oja")
    ax.set_title("Índice de Oja frente al puntaje de PC1 (con los 5 mejores y los 5 peores)")
    ax.legend(title="tópico dominante", fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    save_figure(fig, path)


# --------------------------------------------------------------------------- run


def run(ctx: RunContext) -> dict:
    params = ctx.params
    cfg = oja_config_from_params(params)
    norm_tol = float(params["norm_tol"])
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
    norms = result.history["w_norm"]
    converge = convergence_epoch(norms, norm_tol)

    summary: dict[str, Any] = {
        "input": "features",
        "n_samples": int(Z.shape[0]),
        "n_features": int(Z.shape[1]),
        "epochs": cfg.epochs,
        "eta0": cfg.eta0,
        "eta_min": cfg.eta_min,
        "eta_decay": cfg.eta_decay,
        "norm_tol": norm_tol,
        "init": "uniform01",
        "diverged": bool(result.diverged),
        "divergence_epoch": result.divergence_epoch,
        "initial_w_norm": round(float(np.linalg.norm(result.initial_weights)), 4),
        "backend_resolved": ctx.backend.name,
    }
    if result.diverged:
        if ctx.plots:
            _plot_convergence(result.history, norm_tol, None, result.divergence_epoch, ctx.artifact("convergence.png"))
        summary.update(
            {
                "final_w_norm": None,
                "final_delta_w": None,
                "epochs_to_converge": None,
                "cosine_to_pc1": None,
                "pearson_scores_pc1": None,
                "orientation_matches_pc1": None,
                "orientation_flipped": None,
                "top_creator": None,
                "bottom_creator": None,
            }
        )
        return summary

    index, w_hat = fit.index, fit.w_hat
    order = rank_order(index)
    topics = dominant_topic(design.raw)
    ids = design.ids.reset_index(drop=True)
    names_all = ids["name"].to_numpy()
    raw = design.raw
    art = reconstruct_topic_art(raw).to_numpy()
    evidence = som_evidence(load_som_config(ctx.config["som_config"], who="oja"), Z, xp=ctx.backend.xp)

    ranking = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "creator_id": ids["creator_id"].to_numpy()[order],
            "name": names_all[order],
            "dominant_topic": topics.to_numpy()[order],
            "topic_football_pct": raw["topic_football_pct"].to_numpy()[order],
            "topic_makeup_pct": raw["topic_makeup_pct"].to_numpy()[order],
            "topic_dogs_pct": raw["topic_dogs_pct"].to_numpy()[order],
            "topic_art_pct": art[order],
            "followers": raw["followers"].to_numpy()[order],
            "oja_index": index[order],
            "pc1_score": pca.scores[order, 0],
            "som_region": evidence.region[order],
        }
    )
    write_csv(ctx.artifact("ranking.csv"), ranking)

    weights = pd.DataFrame(
        {
            "feature": features,
            "oja_weight": w_hat,
            "pc1_loading": pc1,
            "difference": w_hat - pc1,
            "oja_weight_raw": result.weights,
        }
    )
    write_csv(ctx.artifact("weights_vs_pc1.csv"), weights)

    if ctx.plots:
        _plot_convergence(result.history, norm_tol, converge, None, ctx.artifact("convergence.png"))
        _plot_cosine(result.history, ctx.artifact("cos_to_pc1.png"))
        _plot_weights(weights, ctx.artifact("weights_vs_pc1.png"))
        _plot_scores(index, pca.scores[:, 0], names_all, topics.to_numpy(), order, ctx.artifact("scores_vs_pc1.png"))

    names = ranking["name"].tolist()
    topic_of = ranking["dominant_topic"].tolist()
    lambda1 = float(pca.eigenvalues[0])
    summary.update(
        {
            "final_w_norm": round(float(norms[-1]), 4),
            "final_delta_w": round(float(result.history["delta_w"][-1]), 6),
            "epochs_to_converge": converge,
            "cosine_to_pc1": round(min(1.0, abs(float(w_hat @ pc1))), 6),
            "pearson_scores_pc1": round(float(np.corrcoef(index, pca.scores[:, 0])[0, 1]), 6),
            "orientation_matches_pc1": bool(w_hat @ pc1 > 0),
            "orientation_flipped": fit.flipped,
            "oja_top_feature": _top_feature(w_hat, features),
            "pc1_top_feature": _top_feature(pc1, features),
            "max_abs_weight_diff": round(float(np.max(np.abs(w_hat - pc1))), 6),
            "lambda1": round(lambda1, 4),
            "course_eta_bound": round(1.0 / (1.2 * lambda1), 4),
            "top_creator": names[0],
            "top_creator_topic": topic_of[0],
            "bottom_creator": names[-1],
            "bottom_creator_topic": topic_of[-1],
            "top5_creators": names[:5],
            "bottom5_creators": names[-5:],
            "som_config": str(ctx.config["som_config"]),
            "n_regions": int(evidence.n_regions),
        }
    )
    return summary


EXPERIMENT = Experiment(
    kind="oja",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
    sweep_report=report,
)
