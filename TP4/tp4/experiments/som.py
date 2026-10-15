"""Experiment kind `som`: hand-written Kohonen map on the 25 standardized selected features.

Phase 1 decided (docs/hallazgos/ej1.md) that Kohonen trains on the standardized
selected variables, not on principal components.

Artifacts:
  bmus.csv      one row per creator (CSV order): winning neuron, grid cell, distance, second winner
  neurons.csv   one row per neuron: grid cell, position, hits, creators mapped there
  weights.csv   trained weight vector of every neuron (standardized units)
  hit_map.png   hit map with every creator's name drawn in its winning neuron
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patheffects import withStroke

from ..analysis.pca import standardize
from ..errors import ConfigError
from ..features import dominant_topic, load_design
from ..models.som import ADJACENCY_TOL, SomConfig, best_two_units, train_som
from ..plotting import draw_grid_cells, plt, save_figure
from ..runs import Experiment, RunContext, write_csv
from .pca import TOPIC_COLORS, TOPIC_LABELS

EXTRA_KEYS = frozenset({"data", "feature_set"})
PARAM_DEFAULTS: dict[str, Any] = {
    "rows": 6,
    "cols": None,
    "epochs": 200,
    "eta0": 0.5,
    "eta_min": 0.01,
    "radius0": 3.0,
    "radius_min": 1.0,
}
MAX_NEURONS = 400
MAX_EPOCHS = 5000


def _is_int(v: Any) -> bool:
    return type(v) is int


def _is_real(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def _grid_dims(p: dict) -> tuple[int, int]:
    return p["rows"], (p["cols"] if p["cols"] is not None else p["rows"])


def validate(config: dict) -> None:
    for key in ("data", "feature_set"):
        v = config.get(key)
        if not isinstance(v, str) or not v:
            raise ConfigError(f"som: {key} must be a non-empty string path, got {v!r}")
    p = config["params"]
    if not (_is_int(p["rows"]) and p["rows"] >= 1):
        raise ConfigError(f"som: params.rows must be an integer >= 1, got {p['rows']!r}")
    if p["cols"] is not None and not (_is_int(p["cols"]) and p["cols"] >= 1):
        raise ConfigError(f"som: params.cols must be null or an integer >= 1, got {p['cols']!r}")
    rows, cols = _grid_dims(p)
    if rows * cols < 2 or rows * cols > MAX_NEURONS:
        raise ConfigError(
            f"som: the map needs between 2 and {MAX_NEURONS} neurons, got {rows}x{cols} = {rows * cols}"
        )
    if not (_is_int(p["epochs"]) and 1 <= p["epochs"] <= MAX_EPOCHS):
        raise ConfigError(f"som: params.epochs must be an integer in [1, {MAX_EPOCHS}], got {p['epochs']!r}")
    eta0, eta_min = p["eta0"], p["eta_min"]
    if not (_is_real(eta0) and _is_real(eta_min) and 0 < eta_min <= eta0 <= 1):
        raise ConfigError(f"som: need 0 < params.eta_min <= params.eta0 <= 1, got eta_min={eta_min!r}, eta0={eta0!r}")
    r0, rmin = p["radius0"], p["radius_min"]
    if not (_is_real(r0) and _is_real(rmin) and 0 < rmin <= r0):
        raise ConfigError(f"som: need 0 < params.radius_min <= params.radius0, got radius_min={rmin!r}, radius0={r0!r}")


def som_config_from_params(params) -> SomConfig:
    rows, cols = _grid_dims(dict(params))
    return SomConfig(
        rows=int(rows),
        cols=int(cols),
        topology=params.get("topology", "hex"),
        neighborhood=params.get("neighborhood", "gaussian"),
        init=params.get("init", "sample"),
        epochs=int(params["epochs"]),
        eta0=float(params["eta0"]),
        eta_min=float(params["eta_min"]),
        eta_decay=params.get("eta_decay", "exponential"),
        radius0=float(params["radius0"]),
        radius_min=float(params["radius_min"]),
        radius_decay=params.get("radius_decay", "exponential"),
    )


def labels_by_neuron(bmu, names, n_neurons: int) -> list[list[str]]:
    """Names mapped to each neuron, in CSV order (every name lands in exactly one neuron)."""
    out: list[list[str]] = [[] for _ in range(n_neurons)]
    for b, name in zip(bmu, names):
        out[int(b)].append(str(name))
    return out


def _plot_hit_map(grid, hits, labels, creator_topic_by_name, topology, path) -> None:
    rows, cols = grid.rows, grid.cols
    height_unit = math.sqrt(3) / 2 if topology == "hex" else 1.0
    fig, ax = plt.subplots(figsize=(1.6 * cols + 2.5, 1.6 * rows * height_unit + 1.8))
    # vmax above the peak keeps the darkest cell mid-tone so the colored names stay legible
    collection = draw_grid_cells(ax, grid.positions, topology, hits, cmap="Blues", vmin=0, vmax=max(1.8 * float(hits.max()), 1.0))
    cbar = fig.colorbar(collection, ax=ax, shrink=0.7)
    cbar.set_label("creadores por neurona")
    cbar.set_ticks(list(range(0, int(hits.max()) + 1)))
    for (x, y), names in zip(grid.positions, labels):
        k = len(names)
        if k == 0:
            continue
        fontsize = 6 if k <= 3 else 5 if k <= 7 else 4
        step = fontsize * 1.4 / 72 / 1.6  # data units: font height in inches over ~1.6 in per unit
        for j, name in enumerate(names):
            ax.text(
                x, y + (j - (k - 1) / 2) * step, name.replace("$", r"\$"),
                ha="center", va="center", fontsize=fontsize,
                color=TOPIC_COLORS[creator_topic_by_name[name]], zorder=3,
                path_effects=[withStroke(linewidth=1.5, foreground="white")],
            )
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=TOPIC_COLORS[t], label=TOPIC_LABELS[t])
        for t in TOPIC_COLORS
    ]
    ax.legend(handles=handles, title="tópico dominante", loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
    ax.set_title(f"Mapa de impactos: creadores por neurona ({rows}x{cols}, {topology})")
    save_figure(fig, path)


def run(ctx: RunContext) -> dict:
    params = ctx.params
    design = load_design(ctx.config["data"], ctx.config["feature_set"])
    Z, _mean, _std = standardize(design.X)
    cfg = som_config_from_params(params)
    result = train_som(Z, cfg, ctx.rng, xp=ctx.backend.xp)
    grid, W = result.grid, result.weights
    n_neurons = grid.n_neurons

    Zn = Z
    bmu, second, bmu_dist = best_two_units(Zn, W)
    hits = np.bincount(bmu, minlength=n_neurons)
    topics = dominant_topic(design.raw)
    names = design.raw["name"].tolist()

    bmus = pd.concat(
        [
            design.ids.reset_index(drop=True),
            topics.reset_index(drop=True),
            pd.DataFrame(
                {
                    "bmu": bmu,
                    "row": bmu // grid.cols,
                    "col": bmu % grid.cols,
                    "bmu_distance": bmu_dist,
                    "second_bmu": second,
                    "second_adjacent": grid.distances[bmu, second] <= 1.0 + ADJACENCY_TOL,
                }
            ),
        ],
        axis=1,
    )
    write_csv(ctx.artifact("bmus.csv"), bmus)

    labels = labels_by_neuron(bmu, names, n_neurons)
    idx = np.arange(n_neurons)
    neurons = pd.DataFrame(
        {
            "neuron": idx,
            "row": idx // grid.cols,
            "col": idx % grid.cols,
            "x": grid.positions[:, 0],
            "y": grid.positions[:, 1],
            "hits": hits,
            "creators": [";".join(n) for n in labels],
        }
    )
    write_csv(ctx.artifact("neurons.csv"), neurons)

    weights = pd.DataFrame(W, columns=list(design.X.columns))
    weights.insert(0, "col", idx % grid.cols)
    weights.insert(0, "row", idx // grid.cols)
    weights.insert(0, "neuron", idx)
    write_csv(ctx.artifact("weights.csv"), weights)

    if ctx.plots:
        topic_by_name = dict(zip(names, topics.tolist()))
        _plot_hit_map(grid, hits, labels, topic_by_name, cfg.topology, ctx.artifact("hit_map.png"))

    n_empty = int((hits == 0).sum())
    return {
        "input": "features",
        "n_samples": int(Zn.shape[0]),
        "n_features": int(Zn.shape[1]),
        "rows": grid.rows,
        "cols": grid.cols,
        "n_neurons": n_neurons,
        "topology": cfg.topology,
        "epochs": cfg.epochs,
        "iterations": cfg.epochs * int(Zn.shape[0]),
        "eta0": round(cfg.eta0, 4),
        "radius0": round(cfg.radius0, 4),
        "quantization_error": round(float(result.history["quantization_error"][-1]), 4),
        "topographic_error": round(float(result.history["topographic_error"][-1]), 4),
        "n_empty_neurons": n_empty,
        "empty_neuron_frac": round(n_empty / n_neurons, 4),
        "max_hits": int(hits.max()),
        "n_creators_mapped": int(len(bmu)),
        "backend_resolved": ctx.backend.name,
    }


EXPERIMENT = Experiment(
    kind="som",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
