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
  u_matrix.png, u_edges.csv  mean distance to adjacent neurons and the strongest neighbor walls
  prototypes.csv, component_planes.png, pc_planes.png  neuron prototypes (original units) and their PC1/PC2
  region_scan.csv, regions.csv, region_profiles.csv, region_map.png, region_profiles.png, biplot_regions.png
                contiguous map regions (Ward on the prototypes), their creators and profiles
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patheffects import withStroke

from ..analysis.pca import fit_pca, standardize
from ..analysis.som_maps import (
    REGION_K_RANGE,
    choose_n_regions,
    correlation_ratio,
    edge_feature_diffs,
    pc_projection,
    prototypes_original_units,
    reconstruct_topic_art,
    region_border_ratio,
    region_profiles,
    region_scan,
    segment_regions,
    u_edges,
    u_matrix,
)
from ..errors import ConfigError
from ..features import TOPIC_COLUMNS, dominant_topic, load_design
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
from ..plotting import draw_grid_cells, draw_region_borders, grid_shape, plt, save_figure
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
    "n_regions": None,
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
    if p["n_regions"] is not None and not (_is_int(p["n_regions"]) and 1 <= p["n_regions"] <= rows * cols):
        raise ConfigError(
            f"som: params.n_regions must be null (automatic) or an integer in [1, {rows * cols}], got {p['n_regions']!r}"
        )
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


def _draw_names(ax, grid, labels, creator_topic_by_name) -> None:
    """Creator names inside their neuron, colored by dominant topic (shared by hit and region maps)."""
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


def _topic_legend(ax) -> None:
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=TOPIC_COLORS[t], label=TOPIC_LABELS[t])
        for t in TOPIC_COLORS
    ]
    ax.legend(handles=handles, title="tópico dominante", loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)


def _border_pairs(grid, region_labels):
    return [(a, b) for a, b in grid.adjacency_pairs() if region_labels[a] != region_labels[b]]


def _plot_hit_map(grid, hits, labels, creator_topic_by_name, topology, path, border_pairs=()) -> None:
    rows, cols = grid.rows, grid.cols
    height_unit = math.sqrt(3) / 2 if topology == "hex" else 1.0
    fig, ax = plt.subplots(figsize=(1.6 * cols + 2.5, 1.6 * rows * height_unit + 1.8))
    # vmax above the peak keeps the darkest cell mid-tone so the colored names stay legible
    collection = draw_grid_cells(ax, grid.positions, topology, hits, cmap="Blues", vmin=0, vmax=max(1.8 * float(hits.max()), 1.0))
    cbar = fig.colorbar(collection, ax=ax, shrink=0.7)
    cbar.set_label("creadores por neurona")
    cbar.set_ticks(list(range(0, int(hits.max()) + 1)))
    _draw_names(ax, grid, labels, creator_topic_by_name)
    draw_region_borders(ax, grid.positions, topology, border_pairs)
    _topic_legend(ax)
    ax.set_title(f"Mapa de impactos: creadores por neurona ({rows}x{cols}, {topology})")
    save_figure(fig, path)


def _plot_u_matrix(grid, u, hits, topology, path, border_pairs=()) -> None:
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
    draw_region_borders(ax, grid.positions, topology, border_pairs, color="#e41a1c", linewidth=2.6)
    ax.set_title(f"Matriz U ({rows}x{cols}, {topology}): distancias altas = fronteras entre grupos")
    save_figure(fig, path)


def _region_legend_handles(counts, colors):
    return [
        plt.Line2D([], [], marker="s", markersize=9, linestyle="", color=colors[r], markeredgecolor="gray",
                   label=f"región {r + 1} ({int(counts[r])} creadores)")
        for r in range(len(counts))
    ]


def _plot_region_map(grid, region_labels, counts, labels, creator_topic_by_name, topology, path) -> None:
    rows, cols = grid.rows, grid.cols
    height_unit = math.sqrt(3) / 2 if topology == "hex" else 1.0
    fig, ax = plt.subplots(figsize=(1.6 * cols + 4.0, 1.6 * rows * height_unit + 1.8))
    cmap = plt.get_cmap("Pastel1")
    colors = [cmap((r % cmap.N)) for r in range(len(counts))]
    draw_grid_cells(ax, grid.positions, topology, colors=[colors[int(r) - 1] for r in region_labels])
    _draw_names(ax, grid, labels, creator_topic_by_name)
    draw_region_borders(ax, grid.positions, topology, _border_pairs(grid, region_labels))
    _topic_legend(ax)
    ax.add_artist(ax.get_legend())
    ax.legend(handles=_region_legend_handles(counts, colors), loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
              ncol=1 if len(counts) <= 20 else 2, fontsize=8)
    ax.set_title(f"Regiones contiguas del mapa ({rows}x{cols}, {topology}): {len(counts)} regiones (Ward sobre los prototipos)")
    save_figure(fig, path)


def _plot_region_profiles(profile, k, counts, path) -> None:
    features = list(dict.fromkeys(profile["feature"]))
    matrix = profile.pivot(index="region", columns="feature", values="mean_z").reindex(columns=features)
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    bound = max(float(np.abs(finite).max()) if finite.size else 0.0, 1e-9)
    fig, ax = plt.subplots(figsize=(0.62 * len(features) + 3.0, 0.55 * k + 2.8))
    image = ax.imshow(np.ma.masked_invalid(values), cmap="vlag", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(features)), labels=features, rotation=90, fontsize=8)
    ax.set_yticks(range(k), labels=[f"región {r + 1} ({int(counts[r])})" for r in range(k)], fontsize=8)
    for i in range(k):
        for j in range(len(features)):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(values[i, j]) > 0.6 * bound else "black")
    fig.colorbar(image, ax=ax, shrink=0.8, label="media z de los creadores de la región")
    ax.set_title("Perfil de cada región: media z de sus creadores")
    fig.tight_layout()
    save_figure(fig, path)


def _plot_biplot_regions(creator_pcs, creator_regions, names, counts, explained_pct, path) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))
    cmap = plt.get_cmap("Dark2")
    for r in range(len(counts)):
        mask = creator_regions == r + 1
        if mask.any():
            ax.scatter(
                creator_pcs[mask, 0], creator_pcs[mask, 1], s=45, color=cmap(r % cmap.N), edgecolor="white",
                linewidth=0.5, zorder=3, label=f"región {r + 1} ({int(counts[r])} creadores)",
            )
    for (x, y), name in zip(creator_pcs, names):
        ax.annotate(str(name), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6, zorder=4)
    ax.axhline(0, color="lightgray", linewidth=0.8)
    ax.axvline(0, color="lightgray", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({explained_pct[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_pct[1]:.1f}%)")
    ax.set_title("Regiones del mapa de Kohonen sobre el plano PC1-PC2")
    ax.legend(title="región del mapa", loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    save_figure(fig, path)


def _plot_component_planes(grid, prototypes, topology, path) -> None:
    n_panels = prototypes.shape[1]
    nrows, ncols = grid_shape(n_panels, 6)
    height_unit = math.sqrt(3) / 2 if topology == "hex" else 1.0
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, (0.38 * grid.rows * height_unit + 0.9) * nrows), squeeze=False
    )
    flat = list(axes.ravel())
    for ax, column in zip(flat, prototypes.columns):
        collection = draw_grid_cells(ax, grid.positions, topology, prototypes[column].to_numpy(), cmap="viridis")
        cbar = fig.colorbar(collection, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
        title = "topic_art_pct (reconstruida)" if column == "topic_art_pct_rec" else column
        ax.set_title(title, fontsize=8)
    for ax in flat[n_panels:]:
        ax.axis("off")
    fig.suptitle(
        "Planos de componentes: valor de cada variable en el prototipo de cada neurona (unidades originales)",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, path)


def _plot_pc_planes(grid, neuron_pcs, explained_pct, topology, path) -> None:
    height_unit = math.sqrt(3) / 2 if topology == "hex" else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(2 * (0.9 * grid.cols + 2.2), 0.9 * grid.rows * height_unit + 1.8))
    for i, ax in enumerate(axes):
        values = neuron_pcs[:, i]
        bound = max(float(np.abs(values).max()), 1e-9)
        collection = draw_grid_cells(ax, grid.positions, topology, values, cmap="vlag", vmin=-bound, vmax=bound)
        fig.colorbar(collection, ax=ax, shrink=0.7)
        ax.set_title(f"PC{i + 1} ({explained_pct[i]:.1f}%) del prototipo de cada neurona", fontsize=10)
    fig.tight_layout()
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


def _regions_table(k, region_labels, creator_regions, names, topics, profile, creator_pcs) -> pd.DataFrame:
    """One row per region: sizes, creators, topic mix, most extreme features and PC means (blank when empty)."""
    topic_names = [c.removeprefix("topic_").removesuffix("_pct") for c in TOPIC_COLUMNS]
    topic_values = topics.to_numpy()
    rows = []
    for r in range(1, k + 1):
        mask = creator_regions == r
        n = int(mask.sum())
        row = {
            "region": r,
            "n_neurons": int((region_labels == r).sum()),
            "n_creators": n,
            "creators": ";".join(name for name, m in zip(names, mask) if m),
            "topic_counts": ";".join(f"{t}:{int((topic_values[mask] == t).sum())}" for t in topic_names),
            "top_high": "",
            "top_low": "",
            "pc1_mean": float("nan"),
            "pc2_mean": float("nan"),
        }
        if n:
            ranked = profile[profile["region"] == r][["feature", "mean_z"]].to_records(index=False)
            ranked = sorted(((str(f), float(z)) for f, z in ranked), key=lambda t: (-t[1], t[0]))
            lows = sorted(ranked, key=lambda t: (t[1], t[0]))
            row["top_high"] = ";".join(f"{f}:{z:+.1f}" for f, z in ranked[:3])
            row["top_low"] = ";".join(f"{f}:{z:+.1f}" for f, z in lows[:3])
            row["pc1_mean"] = float(creator_pcs[mask, 0].mean())
            row["pc2_mean"] = float(creator_pcs[mask, 1].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run(ctx: RunContext) -> dict:
    params = ctx.params
    design = load_design(ctx.config["data"], ctx.config["feature_set"])
    Z, mean, std = standardize(design.X)
    cfg = som_config_from_params(params)
    result = train_som(Z, cfg, ctx.rng, xp=ctx.backend.xp)
    grid, W = result.grid, result.weights
    n_neurons = grid.n_neurons

    # Map readings below never touch ctx.rng: training above is the only consumer.
    feature_names = list(design.X.columns)
    pca = fit_pca(Z, feature_names)  # same oriented components as runs/pca (full SVD, deterministic)
    neuron_pcs = pc_projection(W, pca.components, 2)
    creator_pcs = pca.scores[:, :2]

    Zn = Z
    bmu, second, bmu_dist = best_two_units(Zn, W)
    hits = np.bincount(bmu, minlength=n_neurons)
    topics = dominant_topic(design.raw)
    names = design.raw["name"].tolist()

    # Regions: contiguous groups of neurons (Ward on the prototypes), k from the config or the recorded rule.
    k_values = list(range(REGION_K_RANGE[0], min(REGION_K_RANGE[1], n_neurons - 1) + 1))
    scan = region_scan(W, grid, bmu, k_values)
    n_regions_auto = params["n_regions"] is None
    k = choose_n_regions(scan) if n_regions_auto else int(params["n_regions"])
    region_labels = segment_regions(W, grid, k)
    creator_regions = region_labels[bmu]
    region_counts = np.bincount(creator_regions, minlength=k + 1)[1:]
    border_pairs = _border_pairs(grid, region_labels)

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
                    "pc1": creator_pcs[:, 0],
                    "pc2": creator_pcs[:, 1],
                    "region": creator_regions,
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
            "pc1": neuron_pcs[:, 0],
            "pc2": neuron_pcs[:, 1],
            "region": region_labels,
        }
    )
    write_csv(ctx.artifact("neurons.csv"), neurons)

    prototypes = prototypes_original_units(W, mean, std, feature_names)
    prototypes["topic_art_pct_rec"] = reconstruct_topic_art(prototypes).to_numpy()
    proto_table = prototypes.copy()
    proto_table.insert(0, "col", idx % grid.cols)
    proto_table.insert(0, "row", idx // grid.cols)
    proto_table.insert(0, "neuron", idx)
    write_csv(ctx.artifact("prototypes.csv"), proto_table)

    edges = u_edges(W, grid)
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

    write_csv(ctx.artifact("region_scan.csv"), scan)
    profile = region_profiles(pd.DataFrame(Z, columns=feature_names), design.raw, creator_regions, k)
    write_csv(ctx.artifact("region_profiles.csv"), profile)
    regions = _regions_table(
        k, region_labels, creator_regions, names, topics, profile, creator_pcs
    )
    write_csv(ctx.artifact("regions.csv"), regions)

    preservation = distance_preservation(Zn, W, grid)
    explained_pct = [100.0 * float(r) for r in pca.explained_variance_ratio[:2]]
    if ctx.plots:
        _plot_training_curve(result.history, ctx.artifact("training_curve.png"))
        topic_by_name = dict(zip(names, topics.tolist()))
        _plot_hit_map(grid, hits, labels, topic_by_name, cfg.topology, ctx.artifact("hit_map.png"), border_pairs)
        _plot_u_matrix(grid, u, hits, cfg.topology, ctx.artifact("u_matrix.png"), border_pairs)
        _plot_component_planes(grid, prototypes, cfg.topology, ctx.artifact("component_planes.png"))
        _plot_pc_planes(grid, neuron_pcs, explained_pct, cfg.topology, ctx.artifact("pc_planes.png"))
        _plot_region_map(
            grid, region_labels, region_counts, labels, topic_by_name, cfg.topology, ctx.artifact("region_map.png")
        )
        _plot_region_profiles(profile, k, region_counts, ctx.artifact("region_profiles.png"))
        _plot_biplot_regions(
            creator_pcs, creator_regions, names, region_counts, explained_pct, ctx.artifact("biplot_regions.png")
        )

    n_empty = int((hits == 0).sum())
    ratio = region_border_ratio(W, grid, region_labels)
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
        "pc1_explained_variance_pct": round(float(pca.explained_variance_ratio[0]) * 100.0, 1),
        "pc2_explained_variance_pct": round(float(pca.explained_variance_ratio[1]) * 100.0, 1),
        "n_regions": k,
        "n_regions_auto": n_regions_auto,
        "region_creator_counts": [int(c) for c in region_counts],
        "u_border_ratio": None if math.isnan(ratio) else round(ratio, 4),
        "pc1_region_eta2": round(correlation_ratio(creator_pcs[:, 0], creator_regions), 4),
        "pc2_region_eta2": round(correlation_ratio(creator_pcs[:, 1], creator_regions), 4),
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
