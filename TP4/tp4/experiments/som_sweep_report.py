"""Sweep report for kind `som`: tabulates, plots and selects the Kohonen configuration (SOM-02).

`report(ctx, table)` is registered as `Experiment.sweep_report` of the `som` kind; the generic
sweep calls it after writing sweep.csv. It applies `tp4.analysis.som_selection` and writes:

  selection.csv         one row per child: the four metrics, the rule stages, candidate rank, selected
  selected_params.json  {"child": ..., "params": {dotted grid key: value}} with the grid's JSON types
  sweep_metrics.png     every metric against every swept hyperparameter, selected value marked

This module must not import tp4.experiments.som or the registry (the som module imports it).
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.som_selection import (
    DP_SLACK,
    MAX_EMPTY_FRAC,
    METRIC_COLUMNS,
    TE_SLACK,
    rule_description,
    select_configuration,
)
from ..plotting import plt, save_figure
from ..runs import RunContext, write_csv, write_json

METRIC_LABELS = {
    "quantization_error": "error de cuantización",
    "topographic_error": "error topográfico",
    "distance_preservation": "preservación de distancias",
    "empty_neuron_frac": "fracción de neuronas vacías",
}
SELECTION_COLUMNS = (*METRIC_COLUMNS, "n_neurons", "epochs", "stage_occupancy", "stage_topology", "stage_order", "candidate_rank", "selected")


def _py(value: Any) -> Any:
    """numpy scalar / NaN to a plain JSON-safe Python value (NaN becomes None)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _leaf(dotted: str) -> str:
    return dotted.rsplit(".", 1)[-1]


def _label(key: str) -> str:
    return _leaf(key)


def _plot_metrics(table: pd.DataFrame, grid: dict[str, list], selected_values: dict[str, Any], path: Path) -> None:
    metrics = list(METRIC_COLUMNS)
    keys = list(grid)
    fig, axes = plt.subplots(
        len(metrics), len(keys), figsize=(3.4 * len(keys), 2.6 * len(metrics)), squeeze=False, sharey="row"
    )
    for r, metric in enumerate(metrics):
        values = pd.to_numeric(table[metric], errors="coerce")
        for c, key in enumerate(keys):
            ax = axes[r][c]
            xs = np.arange(len(grid[key]))
            means, lows, highs = [], [], []
            for v in grid[key]:
                column = table[key].to_numpy()
                sub = values[np.array([x == v for x in column])]
                finite = sub.dropna()
                if len(finite):
                    means.append(float(finite.mean()))
                    lows.append(float(finite.min()))
                    highs.append(float(finite.max()))
                else:
                    means.append(np.nan)
                    lows.append(np.nan)
                    highs.append(np.nan)
            means_a, lows_a, highs_a = map(np.array, (means, lows, highs))
            ax.errorbar(
                xs, means_a, yerr=[means_a - lows_a, highs_a - means_a],
                fmt="o-", color="#0072B2", ecolor="#9ab9d6", capsize=3, linewidth=1.4,
            )
            sel = grid[key].index(selected_values[key])
            ax.axvline(sel, color="#D55E00", linestyle="--", linewidth=1.2)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(v) for v in grid[key]], fontsize=8)
            ax.grid(alpha=0.3)
            if r == len(metrics) - 1:
                ax.set_xlabel(_label(key))
            if c == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
    fig.suptitle("Barrido de hiperparámetros del mapa de Kohonen", fontsize=13)
    fig.text(0.5, 0.0, "media sobre las demás variables, barra = mínimo-máximo; línea punteada = valor elegido", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save_figure(fig, path)


def report(ctx: RunContext, table: pd.DataFrame) -> dict:
    sel = select_configuration(table)
    grid: dict[str, list] = dict(ctx.config["grid"])
    keys = list(grid)

    out = sel.table
    selection = pd.concat([out[["child", *keys]], out[list(SELECTION_COLUMNS)]], axis=1)
    write_csv(ctx.artifact("selection.csv"), selection)

    # Values come from the grid itself, never from DataFrame cells, so int/float/str survive.
    combos = list(itertools.product(*(grid[k] for k in keys)))
    chosen = dict(zip(keys, combos[sel.selected_position]))
    child = str(out.loc[sel.selected_position, "child"])
    write_json(ctx.artifact("selected_params.json"), {"child": child, "params": chosen})

    if ctx.plots:
        _plot_metrics(out, grid, chosen, ctx.artifact("sweep_metrics.png"))

    row = out.loc[sel.selected_position]
    summary: dict[str, Any] = {"selected_child": child}
    for key, value in chosen.items():
        summary[f"selected_{_leaf(key)}"] = value
    for metric in METRIC_COLUMNS:
        summary[f"selected_{metric}"] = _py(row[metric])
    summary.update(
        {
            "n_occupancy_ok": int(out["stage_occupancy"].sum()),
            "n_topology_ok": int(out["stage_topology"].sum()),
            "n_order_ok": int(out["stage_order"].sum()),
            "occupancy_fallback": sel.occupancy_fallback,
            "rule_max_empty_frac": MAX_EMPTY_FRAC,
            "rule_te_slack": TE_SLACK,
            "rule_dp_slack": DP_SLACK,
            "selection_rule": rule_description(),
        }
    )
    return summary
