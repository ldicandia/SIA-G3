"""Experiment kind `som`: hand-written Kohonen map on the 25 standardized selected features.

Phase 1 decided (docs/hallazgos/ej1.md) that Kohonen trains on the standardized
selected variables, not on principal components.

Artifacts:
  bmus.csv      one row per creator (CSV order): winning neuron, grid cell, distance, second winner
  neurons.csv   one row per neuron: grid cell, position, hits, creators mapped there
  weights.csv   trained weight vector of every neuron (standardized units)
  history.csv   per epoch: eta, radius, quantization error, topographic error
  hit_map.png   hit map with every creator's name drawn in its winning neuron
  training_curve.png  errors and schedules per epoch
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patheffects import withStroke

from ..analysis.pca import standardize
from ..analysis.som_maps import edge_feature_diffs, u_edges, u_matrix
from ..errors import ConfigError
from ..features import dominant_topic, load_design
from ..models.som import (
    ADJACENCY_TOL,
    DECAYS,
    INITS,
    NEIGHBORHOODS,
    TOPOLOGIES,
    SomConfig,
    best_two_units,
    distance_preservation,
    train_som,
)
from ..plotting import draw_grid_cells, plt, save_figure
from ..runs import Experiment, RunContext, write_csv
from .pca import TOPIC_COLORS, TOPIC_LABELS
from .som_sweep_report import report

EXTRA_KEYS = frozenset({"data", "feature_set"})
PARAM_DEFAULTS: dict[str, Any] = {
    "rows": 6,
    "cols": None,
    "topology": "hex",
    "neighborhood": "gaussian",
    "init": "sample",
    "epochs": 200,
    "eta0": 0.5,
    "eta_min": 0.01,
    "eta_decay": "exponential",
    "radius0": 3.0,
    "radius_min": 1.0,
    "radius_decay": "exponential",
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
    for key, allowed in (
        ("topology", TOPOLOGIES),
        ("neighborhood", NEIGHBORHOODS),
        ("init", INITS),
        ("eta_decay", DECAYS),
        ("radius_decay", DECAYS),
    ):
        if p[key] not in allowed:
            raise ConfigError(f"som: params.{key} must be one of {list(allowed)}, got {p[key]!r}")
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


def _plot_u_matrix(grid, u, hits, topology, path) -> None:
    rows, cols = grid.rows, grid.cols
    height_unit = math.sqrt(3) / 2 if topology == "hex" else 1.0
    fig, ax = plt.subplots(figsize=(1.4 * cols + 2.5, 1.4 * rows * height_unit + 1.5))
    vmin, vmax = float(np.nanmin(u)), float(np.nanmax(u))
    collection = draw_grid_cells(ax, grid.positions, topology, u, cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(collection, ax=ax, shrink=0.7)
    cbar.set_label("distancia media a las neuronas vecinas (unidades z)")
    span = (vmax - vmin) or 1.0
    for (x, y), value, count in zip(grid.positions, u, hits):
        dark = (value - vmin) / span < 0.5  # viridis is dark at the low end
        ax.text(
            x, y, str(int(count)), ha="center", va="center", fontsize=7,
            color="white" if dark else "black", zorder=3,
        )
    ax.set_title(f"Matriz U ({rows}x{cols}, {topology}): distancias altas = fronteras entre grupos")
    save_figure(fig, path)


def _plot_training_curve(history, path) -> None:
    epochs = history["epoch"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    (qe,) = top.plot(epochs, history["quantization_error"], color="#0072B2", label="error de cuantización (QE)")
    top.set_ylabel("QE (distancia media al ganador)")
    top_te = top.twinx()
    (te,) = top_te.plot(epochs, history["topographic_error"], color="#D55E00", label="error topográfico (TE)")
    top_te.set_ylabel("TE (fracción)")
    top.legend(handles=[qe, te], loc="upper right")
    top.grid(alpha=0.3)
    top.set_title("Entrenamiento del mapa: errores y parámetros por época")
    (eta,) = bottom.plot(epochs, history["eta"], color="#009E73", label="tasa de aprendizaje (eta)")
    bottom.set_ylabel("eta")
    bottom_r = bottom.twinx()
    (rad,) = bottom_r.plot(epochs, history["radius"], color="#CC79A7", label="radio de vecindad")
    bottom_r.set_ylabel("radio (unidades de grilla)")
    bottom.legend(handles=[eta, rad], loc="upper right")
    bottom.grid(alpha=0.3)
    bottom.set_xlabel("época")
    fig.tight_layout()
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
    u = u_matrix(W, grid)
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
            "u_distance": u,
        }
    )
    write_csv(ctx.artifact("neurons.csv"), neurons)

    edges = u_edges(W, grid)
    feature_names = list(design.X.columns)
    ea, eb = edges["neuron_a"].to_numpy(), edges["neuron_b"].to_numpy()
    edge_table = pd.DataFrame(
        {
            "neuron_a": ea,
            "row_a": ea // grid.cols,
            "col_a": ea % grid.cols,
            "neuron_b": eb,
            "row_b": eb // grid.cols,
            "col_b": eb % grid.cols,
            "distance": edges["distance"].to_numpy(),
            "creators_a": [";".join(labels[a]) for a in ea],
            "creators_b": [";".join(labels[b]) for b in eb],
            "top_diff": [edge_feature_diffs(W, feature_names, int(a), int(b)) for a, b in zip(ea, eb)],
        }
    )
    write_csv(ctx.artifact("u_edges.csv"), edge_table)

    weights = pd.DataFrame(W, columns=list(design.X.columns))
    weights.insert(0, "col", idx % grid.cols)
    weights.insert(0, "row", idx // grid.cols)
    weights.insert(0, "neuron", idx)
    write_csv(ctx.artifact("weights.csv"), weights)

    history = pd.DataFrame(result.history)
    write_csv(ctx.artifact("history.csv"), history)

    preservation = distance_preservation(Zn, W, grid)
    if ctx.plots:
        _plot_training_curve(result.history, ctx.artifact("training_curve.png"))
        topic_by_name = dict(zip(names, topics.tolist()))
        _plot_hit_map(grid, hits, labels, topic_by_name, cfg.topology, ctx.artifact("hit_map.png"))
        _plot_u_matrix(grid, u, hits, cfg.topology, ctx.artifact("u_matrix.png"))

    n_empty = int((hits == 0).sum())
    return {
        "input": "features",
        "n_samples": int(Zn.shape[0]),
        "n_features": int(Zn.shape[1]),
        "rows": grid.rows,
        "cols": grid.cols,
        "n_neurons": n_neurons,
        "topology": cfg.topology,
        "neighborhood": cfg.neighborhood,
        "init": cfg.init,
        "epochs": cfg.epochs,
        "iterations": cfg.epochs * int(Zn.shape[0]),
        "eta0": round(cfg.eta0, 4),
        "eta_min": round(cfg.eta_min, 4),
        "eta_decay": cfg.eta_decay,
        "radius0": round(cfg.radius0, 4),
        "radius_min": round(cfg.radius_min, 4),
        "radius_decay": cfg.radius_decay,
        "quantization_error": round(float(result.history["quantization_error"][-1]), 4),
        "topographic_error": round(float(result.history["topographic_error"][-1]), 4),
        "distance_preservation": None if math.isnan(preservation) else round(preservation, 4),
        "n_empty_neurons": n_empty,
        "empty_neuron_frac": round(n_empty / n_neurons, 4),
        "max_hits": int(hits.max()),
        "mean_hits_occupied": round(int(Zn.shape[0]) / (n_neurons - n_empty), 4),
        "n_creators_mapped": int(len(bmu)),
        "u_mean": round(float(np.nanmean(u)), 4),
        "u_max": round(float(np.nanmax(u)), 4),
        "u_max_edge": f"{int(edge_table['neuron_a'].iloc[0])}~{int(edge_table['neuron_b'].iloc[0])}",
        "backend_resolved": ctx.backend.name,
    }


EXPERIMENT = Experiment(
    kind="som",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
    sweep_report=report,
)
