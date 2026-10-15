"""Experiment kind `pca`: sklearn PCA on the declared (standardized) feature set.

Artifacts:
  explained_variance.csv  one row per component: eigenvalue, ratio, cumulative ratio
  scree.png               per-PC and cumulative explained variance with threshold lines
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..analysis.pca import (
    PcaResult,
    center,
    components_for_threshold,
    fit_pca,
    kaiser_count,
    standardize,
)
from ..errors import ConfigError
from ..features import load_design
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

    if ctx.plots:
        _plot_scree(result, thresholds, ctx.artifact("scree.png"))

    ratio = result.explained_variance_ratio
    summary: dict[str, Any] = {
        "n_samples": result.n_samples,
        "n_features": len(features),
        "standardized": do_standardize,
        "pc1_explained_variance_pct": _pct(ratio[0]),
        "pc2_explained_variance_pct": _pct(ratio[1]) if k > 1 else None,
        "cumulative_pc2_pct": _pct(result.cumulative_ratio[min(1, k - 1)]),
    }
    for t in thresholds:
        summary[f"k_cum_{round(t * 100)}"] = components_for_threshold(result.cumulative_ratio, t)
    summary["kaiser_k"] = kaiser_count(result) if do_standardize else None
    return summary


EXPERIMENT = Experiment(
    kind="pca",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
