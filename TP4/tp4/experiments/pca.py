"""Experiment kind `pca`: sklearn PCA on the declared (standardized) feature set.

Artifacts:
  explained_variance.csv  one row per component: eigenvalue, ratio, cumulative ratio
  scree.png               per-PC and cumulative explained variance with threshold lines
  loadings.csv            oriented eigenvector coefficients, one row per selected feature
  loadings_heatmap.png    features x first `loadings_pcs` components
  scores.csv              one row per creator (CSV order): ids, dominant topic, PC1..PCk
  biplot.png              PC1 vs PC2, every creator labeled, loading arrows
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns

from ..analysis.pca import (
    PcaResult,
    center,
    components_for_threshold,
    eigen_crosscheck,
    fit_pca,
    kaiser_count,
    loadings_frame,
    standardize,
    top_loadings,
)
from ..errors import ConfigError
from ..features import dominant_topic, load_design
from ..plotting import plt, save_figure
from ..runs import Experiment, RunContext, write_csv

EXTRA_KEYS = frozenset({"data", "feature_set"})
PARAM_DEFAULTS: dict[str, Any] = {
    "standardize": True,
    "n_components": None,
    "variance_thresholds": [0.8, 0.9],
    "loadings_pcs": 6,
    "biplot_max_arrows": None,
    "top_n": 5,
}


def _is_int(v: Any) -> bool:
    return type(v) is int


def _is_real(v: Any) -> bool:
    return type(v) in (int, float)


def validate(config: dict) -> None:
    for key in ("data", "feature_set"):
        v = config.get(key)
        if not isinstance(v, str) or not v:
            raise ConfigError(f"pca: {key} must be a non-empty string path, got {v!r}")
    p = config["params"]
    n = p["n_components"]
    if n is not None and not (_is_int(n) and n >= 1):
        raise ConfigError(f"pca: params.n_components must be null or an integer >= 1, got {n!r}")
    th = p["variance_thresholds"]
    if not th or not all(_is_real(t) and 0 < t <= 1 for t in th):
        raise ConfigError(
            f"pca: params.variance_thresholds must be a non-empty list of numbers in (0, 1], got {th!r}"
        )
    for key in ("loadings_pcs", "top_n"):
        if not (_is_int(p[key]) and p[key] >= 1):
            raise ConfigError(f"pca: params.{key} must be an integer >= 1, got {p[key]!r}")
    arrows = p["biplot_max_arrows"]
    if arrows is not None and not (_is_int(arrows) and arrows >= 1):
        raise ConfigError(f"pca: params.biplot_max_arrows must be null or an integer >= 1, got {arrows!r}")


def _component_names(k: int) -> list[str]:
    return [f"PC{i + 1}" for i in range(k)]


def _pct(x: float) -> float:
    return round(float(x) * 100.0, 1)


def _plot_scree(result: PcaResult, thresholds: list[float], path) -> None:
    k = len(result.explained_variance_ratio)
    x = np.arange(1, k + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, result.explained_variance_ratio * 100, color="#4C72B0", label="varianza explicada por componente")
    ax.plot(x, result.cumulative_ratio * 100, color="#C44E52", marker="o", markersize=4, label="varianza acumulada")
    for t in thresholds:
        n_k = components_for_threshold(result.cumulative_ratio, t)
        ax.axhline(t * 100, color="gray", linestyle="--", linewidth=1)
        ax.annotate(
            f"{t * 100:g}%: {n_k} componentes",
            xy=(k, t * 100),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xlabel("componente principal")
    ax.set_ylabel("varianza explicada (%)")
    ax.set_title("Gráfico de sedimentación (scree plot) y varianza explicada acumulada")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    save_figure(fig, path)


TOPIC_COLORS = {"football": "#0072B2", "makeup": "#CC79A7", "dogs": "#E69F00", "art": "#009E73"}
TOPIC_LABELS = {"football": "fútbol", "makeup": "maquillaje", "dogs": "perros", "art": "arte"}


def _pc_label(result: PcaResult, i: int) -> str:
    return f"PC{i + 1} ({result.explained_variance_ratio[i] * 100:.1f}%)"


def _plot_loadings_heatmap(result: PcaResult, n_pcs: int, path) -> None:
    frame = loadings_frame(result, n_pcs)
    frame.columns = [_pc_label(result, i) for i in range(frame.shape[1])]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.3 * frame.shape[1] + 4), max(6.0, 0.36 * len(frame) + 2)))
    sns.heatmap(
        frame,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        linewidths=0.3,
        cbar_kws={"label": "coeficiente (autovector unitario)"},
        ax=ax,
    )
    ax.set_title(f"Cargas (loadings) de las primeras {frame.shape[1]} componentes principales")
    ax.set_xlabel("componente principal")
    ax.set_ylabel("variable")
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    save_figure(fig, path)


def _plot_biplot(result: PcaResult, names, topics, max_arrows: int | None, path) -> None:
    scores = result.scores[:, :2]
    coefs = result.components[:2].T  # (n_features, 2)
    scale = 0.9 * np.abs(scores).max() / np.abs(coefs).max()
    fig, ax = plt.subplots(figsize=(14, 11))
    for topic, color in TOPIC_COLORS.items():
        mask = np.asarray(topics) == topic
        if mask.any():
            ax.scatter(
                scores[mask, 0], scores[mask, 1], s=45, color=color, edgecolor="white", linewidth=0.5,
                label=f"{TOPIC_LABELS[topic]} ({int(mask.sum())})", zorder=3,
            )
    for (x, y), name in zip(scores, names):
        ax.annotate(str(name), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6, zorder=4)

    norms = (coefs**2).sum(axis=1)
    order = np.argsort(-norms, kind="stable")
    shown = order if max_arrows is None else order[:max_arrows]
    for i in sorted(shown):
        dx, dy = coefs[i] * scale
        ax.annotate(
            "", xy=(dx, dy), xytext=(0, 0), zorder=2,
            arrowprops={"arrowstyle": "->", "color": "#555555", "alpha": 0.75, "linewidth": 0.9},
        )
        ax.text(dx * 1.04, dy * 1.04, result.features[i], fontsize=7, color="#333333", zorder=2,
                ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top")
    # Annotations do not autoscale: fit the axes to the creators AND every drawn arrow tip (+ label room).
    tips = coefs[shown] * scale if len(shown) else np.zeros((1, 2))
    extent = np.vstack([scores, tips * 1.15])
    lo, hi = extent.min(axis=0), extent.max(axis=0)
    pad = 0.08 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.axhline(0, color="lightgray", linewidth=0.8)
    ax.axvline(0, color="lightgray", linewidth=0.8)
    ax.set_xlabel(_pc_label(result, 0))
    ax.set_ylabel(_pc_label(result, 1))
    ax.set_title(
        "Biplot PC1 vs PC2: creadores (color = tópico dominante) y cargas de las variables\n"
        f"(flechas = coeficientes de las componentes × {scale:.2f}; {len(shown)} de {len(coefs)} variables)",
        fontsize=11,
    )
    ax.legend(title="tópico dominante", loc="best")
    ax.grid(alpha=0.2)
    save_figure(fig, path)


def _top_strings(result: PcaResult, pc: int, n: int) -> list[str]:
    return [f"{feature}:{coef:+.3f}" for feature, coef in top_loadings(result, pc, n)]


def run(ctx: RunContext) -> dict:
    params = ctx.params
    design = load_design(ctx.config["data"], ctx.config["feature_set"])
    features = list(design.X.columns)
    do_standardize = bool(params["standardize"])
    Z = standardize(design.X)[0] if do_standardize else center(design.X)[0]

    n_components = params["n_components"]
    if n_components is not None and n_components > min(Z.shape):
        raise ConfigError(
            f"pca: params.n_components={n_components} exceeds min(n_samples, n_features)={min(Z.shape)}"
        )
    result = fit_pca(Z, features, n_components)
    k = len(result.explained_variance_ratio)
    thresholds = [float(t) for t in params["variance_thresholds"]]

    table = pd.DataFrame(
        {
            "component": _component_names(k),
            "eigenvalue": result.eigenvalues,
            "explained_variance_ratio": result.explained_variance_ratio,
            "cumulative_ratio": result.cumulative_ratio,
        }
    )
    write_csv(ctx.artifact("explained_variance.csv"), table)

    pc_names = _component_names(k)
    loadings = loadings_frame(result)
    loadings.index.name = "feature"
    write_csv(ctx.artifact("loadings.csv"), loadings.reset_index())
    topics = dominant_topic(design.raw)
    scores = pd.concat(
        [
            design.ids.reset_index(drop=True),
            topics.reset_index(drop=True),
            pd.DataFrame(result.scores, columns=pc_names),
        ],
        axis=1,
    )
    write_csv(ctx.artifact("scores.csv"), scores)

    if ctx.plots:
        _plot_scree(result, thresholds, ctx.artifact("scree.png"))
        _plot_loadings_heatmap(result, int(params["loadings_pcs"]), ctx.artifact("loadings_heatmap.png"))
        if k >= 2:
            _plot_biplot(
                result, design.raw["name"].tolist(), topics.tolist(), params["biplot_max_arrows"],
                ctx.artifact("biplot.png"),
            )

    ratio = result.explained_variance_ratio
    summary: dict[str, Any] = {
        "n_samples": result.n_samples,
        "n_features": len(features),
        "standardized": do_standardize,
        "pc1_explained_variance_pct": _pct(ratio[0]),
        "pc2_explained_variance_pct": _pct(ratio[1]) if k > 1 else None,
        "cumulative_pc2_pct": _pct(result.cumulative_ratio[min(1, k - 1)]),
        "backend_resolved": ctx.backend.name,
        "eigen_crosscheck_max_abs_diff": eigen_crosscheck(Z, ratio, ctx.backend.xp),
    }
    for t in thresholds:
        summary[f"k_cum_{round(t * 100)}"] = components_for_threshold(result.cumulative_ratio, t)
    summary["kaiser_k"] = kaiser_count(result) if do_standardize else None
    top_n = int(params["top_n"])
    for pc in (0, 1):
        label = f"pc{pc + 1}"
        if pc < k:
            feature, coef = top_loadings(result, pc, 1)[0]
            summary[f"{label}_top_feature"] = feature
            summary[f"{label}_top_loading"] = round(coef, 4)
            summary[f"{label}_top5"] = _top_strings(result, pc, top_n)
        else:
            summary[f"{label}_top_feature"] = None
            summary[f"{label}_top_loading"] = None
            summary[f"{label}_top5"] = []
    return summary


EXPERIMENT = Experiment(
    kind="pca",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
