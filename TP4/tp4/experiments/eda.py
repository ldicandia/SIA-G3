"""Experiment kind `eda`: exploratory analysis of creadores.csv.

Artifacts (all rows are kept; outliers are only flagged, never dropped):
  columns.csv       one row per CSV column, in CSV order
  outliers.csv      Tukey-fence outliers, by CSV column order then creator_id
  correlation.csv   Pearson matrix over the numeric columns, in CSV order
  identities.csv    the six documented identities: holds on the data? which dropped column breaks it?
  near_redundant.csv  pairs with |r| >= near_redundant_abs_r over all numeric columns (both_selected flag)
  feature_set.json  resolved feature set: selected columns, dropped reasons, validity
  *.png             distributions, standardized boxplots, scale ranges, heatmaps (all / selected)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns

from ..analysis.eda import (
    correlation_matrix,
    describe_columns,
    iqr_outliers,
    tukey_fences,
    zscores,
)
from ..data import load_creators, numeric_columns
from ..errors import ConfigError
from ..features import check_identities, load_feature_set, near_redundant_pairs, selected_columns
from ..plotting import grid_shape, plt, save_figure
from ..runs import Experiment, RunContext, resolve_project_path, write_csv, write_json

PARAM_DEFAULTS: dict[str, Any] = {"outlier_iqr_k": 1.5, "near_redundant_abs_r": 0.95}
EXTRA_KEYS = frozenset({"data", "feature_set"})


def validate(config: dict) -> None:
    for key in ("data", "feature_set"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"eda: {key} must be a non-empty string path, got {value!r}")
    k = config["params"]["outlier_iqr_k"]
    if not k > 0:
        raise ConfigError(f"eda: params.outlier_iqr_k must be > 0, got {k!r}")
    t = config["params"]["near_redundant_abs_r"]
    if not 0 < t <= 1:
        raise ConfigError(f"eda: params.near_redundant_abs_r must be in (0, 1], got {t!r}")


def _strongest_pair(corr: pd.DataFrame) -> tuple[float | None, str | None]:
    """Largest |r| over distinct column pairs; ties broken alphabetically by 'a~b'."""
    cols = list(corr.columns)
    best: tuple[float, str] | None = None
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if pd.isna(r):
                continue
            key = (-round(abs(float(r)), 10), "~".join(sorted((a, b))))
            if best is None or key < best:
                best = key
    if best is None:
        return None, None
    return -best[0], best[1]


def _plot_distributions(df: pd.DataFrame, cols: list[str], k: float, path) -> None:
    nrows, ncols = grid_shape(len(cols), 6)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows), squeeze=False)
    for ax, col in zip(axes.ravel(), cols):
        values = df[col].dropna()
        ax.hist(values, bins=15, color="#4C72B0", edgecolor="white", linewidth=0.4)
        lo, hi = tukey_fences(values, k)
        for fence in (lo, hi):
            if not np.isnan(fence) and values.min() <= fence <= values.max():
                ax.axvline(fence, color="#C44E52", linestyle="--", linewidth=1)
        ax.set_title(col, fontsize=8)
        ax.set_xlabel("valor", fontsize=6)
        ax.set_ylabel("frecuencia", fontsize=6)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    fig.suptitle(
        f"Distribución de cada variable (líneas punteadas: cercas de Tukey, k={k:g})", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, path)


def _plot_boxplots(df: pd.DataFrame, cols: list[str], k: float, path) -> None:
    z = zscores(df, cols)
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.32 * len(cols))))
    ax.boxplot(
        [z[c].dropna() for c in cols],
        orientation="horizontal",
        whis=k,
        tick_labels=cols,
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#C44E52", "markeredgecolor": "#C44E52"},
    )
    ax.invert_yaxis()
    ax.set_title(f"Boxplots de las variables estandarizadas (bigotes a {k:g}·RIC)")
    ax.set_xlabel("valor estandarizado (z-score)")
    ax.set_ylabel("variable")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, path)


def _plot_scale_ranges(df: pd.DataFrame, cols: list[str], path) -> None:
    stds = pd.Series({c: df[c].std(ddof=1) for c in cols}).dropna()
    stds = stds[stds > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.3 * max(len(stds), 1))))
    if stds.empty:
        ax.text(0.5, 0.5, "ninguna variable con desvío positivo", ha="center", va="center")
    else:
        ax.barh(stds.index, stds.to_numpy(), color="#4C72B0")
        ax.set_xscale("log")
        ax.invert_yaxis()
    ax.set_title("Escala de cada variable (desvío estándar, eje logarítmico)")
    ax.set_xlabel("desvío estándar (escala log)")
    ax.set_ylabel("variable")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, path)


def _plot_heatmap(
    corr: pd.DataFrame, path, title: str = "Matriz de correlación entre variables numéricas"
) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.2,
        cbar_kws={"label": "correlación de Pearson"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("variable")
    ax.set_ylabel("variable")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    save_figure(fig, path)


def _rank(df: pd.DataFrame, columns: list[str]) -> int:
    """Rank of the standardized matrix (a missing cell counts as the mean, i.e. 0)."""
    z = zscores(df, columns).fillna(0.0).to_numpy()
    return int(np.linalg.matrix_rank(z))


def run(ctx: RunContext) -> dict:
    k = float(ctx.params["outlier_iqr_k"])
    df = load_creators(resolve_project_path(ctx.config["data"]))
    cols = numeric_columns(df)

    outliers = iqr_outliers(df, cols, k)
    corr = correlation_matrix(df, cols)
    write_csv(ctx.artifact("columns.csv"), describe_columns(df, k))
    write_csv(ctx.artifact("outliers.csv"), outliers)
    write_csv(ctx.artifact("correlation.csv"), corr.rename_axis("column").reset_index())

    # EDA-02: the EDA REPORTS on the declared feature set; select_features enforces it elsewhere.
    fs = load_feature_set(ctx.config["feature_set"])
    selected = selected_columns(df, fs)
    threshold = float(ctx.params["near_redundant_abs_r"])
    identities = check_identities(df, fs)
    pairs = near_redundant_pairs(corr, threshold, selected)
    feature_set_valid = bool((~(identities["holds"] & (identities["broken_by"] == ""))).all())
    write_csv(ctx.artifact("identities.csv"), identities)
    write_csv(ctx.artifact("near_redundant.csv"), pairs)
    write_json(
        ctx.artifact("feature_set.json"),
        {
            "source": fs.source,
            "selected": selected,
            "n_selected": len(selected),
            "dropped": dict(fs.drop),
            "reconstruct": dict(fs.reconstruct),
            "valid": feature_set_valid,
        },
    )

    if ctx.plots:
        _plot_distributions(df, cols, k, ctx.artifact("distributions.png"))
        _plot_boxplots(df, cols, k, ctx.artifact("boxplots_standardized.png"))
        _plot_scale_ranges(df, cols, ctx.artifact("scale_ranges.png"))
        _plot_heatmap(corr, ctx.artifact("correlation_heatmap.png"))
        _plot_heatmap(
            corr.loc[selected, selected],
            ctx.artifact("correlation_heatmap_selected.png"),
            title=f"Matriz de correlación de las {len(selected)} variables seleccionadas",
        )

    max_r, pair = _strongest_pair(corr)
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_numeric_columns": len(cols),
        "n_missing_cells": int(df.isna().sum().sum()),
        "n_outlier_cells": int(len(outliers)),
        "n_creators_with_outliers": int(outliers["creator_id"].nunique()),
        "outlier_iqr_k": k,
        "max_abs_correlation": max_r,
        "max_abs_correlation_pair": pair,
        "n_identities": int(len(identities)),
        "n_identities_holding": int(identities["holds"].sum()),
        "n_near_redundant_pairs": int(len(pairs)),
        "near_redundant_abs_r": threshold,
        "numeric_rank": _rank(df, cols),
        "selected_rank": _rank(df, selected),
        "n_selected_features": len(selected),
        "feature_set_valid": feature_set_valid,
    }


EXPERIMENT = Experiment(
    kind="eda",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
