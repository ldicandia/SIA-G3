"""Kohonen, mapa de impactos, matriz U y perfiles por neurona (clase, pp. 24–37)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patheffects import withStroke

from ..analysis.pca import PcaInput, load_pca_input
from ..analysis.som_maps import edge_feature_diffs, prototypes_original_units, reconstruct_prototypes, u_edges, u_matrix
from ..config import load_config
from ..errors import ConfigError
from ..features import dominant_topic
from ..models.som import DECAYS, INITS, TOPOLOGIES, SomConfig, matching_units, train_som
from ..plotting import draw_grid_cells, grid_shape, plt, save_figure
from ..runs import Experiment, RunContext, resolve_project_path, write_csv
from .pca import TOPIC_COLORS, TOPIC_LABELS
from .som_sweep_report import report

EXTRA_KEYS = frozenset({"pca_config"})
PARAM_DEFAULTS: dict[str, Any] = {
    "n_repeats": 30,
    "variance_threshold": 0.9,
    "rows": 6,
    "cols": None,
    "topology": "hex",
    "init": "sample",
    "epochs": 200,
    "eta0": 0.5,
    "radius0": 3.0,
    "radius_decay": "inverse",
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
    path = config.get("pca_config")
    if not isinstance(path, str) or not path:
        raise ConfigError(f"som: pca_config must be a non-empty string path, got {path!r}")
    p = config["params"]
    if not (_is_real(p["variance_threshold"]) and 0 < p["variance_threshold"] <= 1):
        raise ConfigError("som: variance_threshold must be in (0, 1]")
    pca_input = load_pca_input(path, p["variance_threshold"])
    if pca_input.n_components < 2:
        raise ConfigError("som: retain at least two principal components to draw the PC1-PC2 planes")
    if type(p["n_repeats"]) is not int or not 2 <= p["n_repeats"] <= 100:
        raise ConfigError("som: n_repeats must be an integer in [2, 100]")
    if not (_is_int(p["rows"]) and p["rows"] >= 1):
        raise ConfigError(f"som: params.rows must be an integer >= 1, got {p['rows']!r}")
    if p["cols"] is not None and not (_is_int(p["cols"]) and p["cols"] >= 1):
        raise ConfigError(f"som: params.cols must be null or an integer >= 1, got {p['cols']!r}")
    for key, allowed in (
        ("topology", TOPOLOGIES),
        ("init", INITS),
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
    eta0 = p["eta0"]
    if not (_is_real(eta0) and 0 < eta0 < 1):
        raise ConfigError("som: need 0 < params.eta0 < 1")
    if not (_is_real(p["radius0"]) and p["radius0"] >= 1):
        raise ConfigError("som: radius0 must be finite and at least 1")


def som_config_from_params(params) -> SomConfig:
    rows, cols = _grid_dims(dict(params))
    return SomConfig(
        rows=int(rows),
        cols=int(cols),
        topology=params.get("topology", "hex"),
        init=params.get("init", "sample"),
        epochs=int(params["epochs"]),
        eta0=float(params["eta0"]),
        radius0=float(params["radius0"]),
        radius_decay=params.get("radius_decay", "inverse"),
    )


@dataclass(frozen=True)
class SomEvidence:
    """Winning neuron and coordinates for each creator, in CSV order."""

    bmu: np.ndarray
    row: np.ndarray
    col: np.ndarray


def load_som_config(path: str, who: str) -> dict[str, Any]:
    """Load and validate a `som` config named by another kind (read only through load_config).

    `tp4.runner` is imported only inside the function: the runner imports the experiment
    registry, which imports this module (a top-level import would cycle).
    """
    from .. import runner

    loaded = load_config(resolve_project_path(path))
    if loaded.get("kind") != "som":
        raise ConfigError(f"{who}: som_config must be a config of kind 'som', got kind {loaded.get('kind')!r}")
    return runner.validate_config(loaded)


def som_evidence(som_config: dict[str, Any], pca_input: PcaInput | None = None) -> SomEvidence:
    if pca_input is None:
        pca_input = load_pca_input(som_config["pca_config"], som_config["params"]["variance_threshold"])
    scores = pca_input.scores
    cfg = som_config_from_params(som_config["params"])
    result = train_som(scores, cfg, np.random.default_rng(som_config["seed"]))
    bmu = matching_units(scores, result.weights)[0]
    return SomEvidence(bmu, bmu // cfg.cols, bmu % cfg.cols)


def labels_by_neuron(bmu, names, n_neurons: int) -> list[list[str]]:
    """Names mapped to each neuron, in CSV order (every name lands in exactly one neuron)."""
    out: list[list[str]] = [[] for _ in range(n_neurons)]
    for b, name in zip(bmu, names):
        out[int(b)].append(str(name))
    return out


def _draw_names(ax, grid, labels, creator_topic_by_name) -> None:
    """Creator names inside their neuron, colored by dominant topic."""
    for (x, y), names in zip(grid.positions, labels):
        k = len(names)
        if k == 0:
            continue
        fontsize = 6 if k <= 3 else 5 if k <= 7 else 4
        step = fontsize * 1.4 / 72 / 1.6
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


def _plot_hit_map(grid, hits, labels, creator_topic_by_name, topology, seed, path) -> None:
    rows, cols = grid.rows, grid.cols
    height_unit = math.sqrt(3) / 2 if topology == "hex" else 1.0
    fig, ax = plt.subplots(figsize=(1.6 * cols + 2.5, 1.6 * rows * height_unit + 1.8))
    collection = draw_grid_cells(ax, grid.positions, topology, hits, cmap="Blues", vmin=0, vmax=max(1.8 * float(hits.max()), 1.0))
    cbar = fig.colorbar(collection, ax=ax, shrink=0.7)
    cbar.set_label("creadores por neurona")
    cbar.set_ticks(list(range(0, int(hits.max()) + 1)))
    _draw_names(ax, grid, labels, creator_topic_by_name)
    _topic_legend(ax)
    ax.set_title(f"Mapa de impactos ({rows}x{cols}, {topology}); semilla {seed} (ejemplo)")
    save_figure(fig, path)


def _plot_u_matrix(grid, u, hits, topology, seed, path) -> None:
    rows, cols = grid.rows, grid.cols
    height_unit = math.sqrt(3) / 2 if topology == "hex" else 1.0
    fig, ax = plt.subplots(figsize=(1.4 * cols + 2.5, 1.4 * rows * height_unit + 1.5))
    vmin, vmax = float(np.nanmin(u)), float(np.nanmax(u))
    collection = draw_grid_cells(ax, grid.positions, topology, u, cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(collection, ax=ax, shrink=0.7)
    cbar.set_label("distancia media a las neuronas vecinas (espacio PCA retenido)")
    span = (vmax - vmin) or 1.0
    for (x, y), value, count in zip(grid.positions, u, hits):
        dark = (value - vmin) / span < 0.5
        ax.text(
            x, y, str(int(count)), ha="center", va="center", fontsize=7,
            color="white" if dark else "black", zorder=3,
        )
    ax.set_title(f"Matriz U ({rows}x{cols}, {topology}); semilla {seed} (ejemplo)")
    save_figure(fig, path)


def _plot_component_planes(grid, prototypes, topology, seed, path) -> None:
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
        title = column
        ax.set_title(title, fontsize=8)
    for ax in flat[n_panels:]:
        ax.axis("off")
    fig.suptitle(
        f"Reconstrucción aproximada de prototipos (unidades originales); semilla {seed} (ejemplo)",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, path)


def _plot_pc_planes(grid, neuron_pcs, explained_pct, topology, seed, path) -> None:
    height_unit = math.sqrt(3) / 2 if topology == "hex" else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(2 * (0.9 * grid.cols + 2.2), 0.9 * grid.rows * height_unit + 1.8))
    for i, ax in enumerate(axes):
        values = neuron_pcs[:, i]
        bound = max(float(np.abs(values).max()), 1e-9)
        collection = draw_grid_cells(ax, grid.positions, topology, values, cmap="vlag", vmin=-bound, vmax=bound)
        fig.colorbar(collection, ax=ax, shrink=0.7)
        ax.set_title(f"PC{i + 1} ({explained_pct[i]:.1f}%) del prototipo de cada neurona", fontsize=10)
    fig.suptitle(f"Coordenadas PCA de los prototipos; semilla {seed} (ejemplo)")
    fig.tight_layout()
    save_figure(fig, path)


def _plot_training_curve(history, n_repeats, path) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    epoch = history["epoch"].to_numpy()
    mean = history["mean_bmu_distance_mean"].to_numpy()
    std = history["mean_bmu_distance_std"].to_numpy()
    top.plot(epoch, mean, label="media")
    top.fill_between(epoch, mean - std, mean + std, alpha=0.2, label="±1 desvío entre semillas")
    top.set_ylabel("distancia media al ganador (PCA)")
    top.set_title(f"Kohonen: {n_repeats} entrenamientos independientes")
    top.legend()
    top.grid(alpha=0.3)
    bottom.plot(epoch, history["eta"], label="eta")
    bottom.plot(epoch, history["radius"], label="radio")
    bottom.set_xlabel("época")
    bottom.legend()
    bottom.grid(alpha=0.3)
    save_figure(fig, path)


def _plot_biplot_cells(scores, bmu, names, grid, explained_pct, seed, path) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    points = ax.scatter(scores[:, 0], scores[:, 1], c=bmu, cmap="turbo", s=35)
    for (x, y), cell, name in zip(scores, bmu, names):
        ax.annotate(f"{name} ({cell // grid.cols},{cell % grid.cols})", (x, y), xytext=(3, 3),
                    textcoords="offset points", fontsize=6)
    ax.set_xlabel(f"PC1 ({explained_pct[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_pct[1]:.1f}%)")
    ax.set_title(f"PC1-PC2 y celdas (fila,columna); semilla {seed} (ejemplo)")
    fig.colorbar(points, ax=ax, label="índice de la neurona ganadora")
    ax.grid(alpha=0.2)
    save_figure(fig, path)


REPEAT_METRICS = (
    "mean_bmu_distance", "mean_bmu_distance_original", "n_empty_neurons", "empty_neuron_frac", "max_hits",
    "mean_hits_occupied", "u_mean", "u_max",
)


def repeat_training(scores, cfg, seed, n_repeats, *, original_Z, components):
    records, curves = [], []
    representative = None
    for run_seed in range(seed, seed + n_repeats):
        result = train_som(scores, cfg, np.random.default_rng(run_seed))
        if representative is None:
            representative = result
        bmu, distance = matching_units(scores, result.weights)
        reconstructed = reconstruct_prototypes(result.weights, components)
        original_distance = np.linalg.norm(original_Z - reconstructed[bmu], axis=1)
        hits = np.bincount(bmu, minlength=result.grid.n_neurons)
        n_empty = int((hits == 0).sum())
        u = u_matrix(result.weights, result.grid)
        records.append({
            "seed": run_seed, "mean_bmu_distance": float(distance.mean()),
            "mean_bmu_distance_original": float(original_distance.mean()),
            "n_empty_neurons": n_empty, "empty_neuron_frac": n_empty / len(hits),
            "max_hits": int(hits.max()), "mean_hits_occupied": len(scores) / (len(hits) - n_empty),
            "u_mean": float(np.nanmean(u)), "u_max": float(np.nanmax(u)),
        })
        curves.append(result.history["mean_bmu_distance"])
    values = np.stack(curves)
    history = pd.DataFrame({
        "epoch": representative.history["epoch"],
        "eta": representative.history["eta"], "radius": representative.history["radius"],
        "mean_bmu_distance_mean": values.mean(axis=0),
        "mean_bmu_distance_std": values.std(axis=0, ddof=1),
    })
    return representative, pd.DataFrame(records), history


def run(ctx: RunContext) -> dict:
    params = ctx.params
    pca_input = load_pca_input(ctx.config["pca_config"], params["variance_threshold"])
    design, Z = pca_input.design, pca_input.Z
    pca, scores = pca_input.pca, pca_input.scores
    mean, std = pca_input.mean, pca_input.std
    cfg = som_config_from_params(params)
    result, per_seed, history_repeats = repeat_training(
        scores, cfg, ctx.seed, params["n_repeats"], original_Z=Z, components=pca.components
    )
    write_csv(ctx.artifact("per_seed.csv"), per_seed)
    write_csv(ctx.artifact("history_repeats.csv"), history_repeats)
    grid, W = result.grid, result.weights
    n_neurons = grid.n_neurons

    feature_names = list(design.X.columns)
    neuron_pcs = W[:, :2]
    creator_pcs = pca.scores[:, :2]
    reconstructed = reconstruct_prototypes(W, pca.components)

    bmu, bmu_dist = matching_units(scores, W)
    bmu_dist_original = np.linalg.norm(Z - reconstructed[bmu], axis=1)
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
                    "bmu_distance_original": bmu_dist_original,
                    "pc1": creator_pcs[:, 0],
                    "pc2": creator_pcs[:, 1],
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
        }
    )
    write_csv(ctx.artifact("neurons.csv"), neurons)

    prototypes = prototypes_original_units(reconstructed, mean, std, feature_names)
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
            "top_diff": [edge_feature_diffs(reconstructed, feature_names, int(a), int(b)) for a, b in zip(ea, eb)],
        }
    )
    write_csv(ctx.artifact("u_edges.csv"), edge_table)

    weights = pd.DataFrame(W, columns=[f"PC{i + 1}" for i in range(pca_input.n_components)])
    weights.insert(0, "col", idx % grid.cols)
    weights.insert(0, "row", idx // grid.cols)
    weights.insert(0, "neuron", idx)
    write_csv(ctx.artifact("weights.csv"), weights)
    reconstructed_weights = pd.DataFrame(reconstructed, columns=feature_names)
    reconstructed_weights.insert(0, "col", idx % grid.cols)
    reconstructed_weights.insert(0, "row", idx // grid.cols)
    reconstructed_weights.insert(0, "neuron", idx)
    write_csv(ctx.artifact("reconstructed_weights.csv"), reconstructed_weights)

    history = pd.DataFrame(result.history)
    write_csv(ctx.artifact("history.csv"), history)

    profiles = design.raw.select_dtypes(include="number").copy()
    profiles["bmu"] = bmu
    profiles["pc1"] = creator_pcs[:, 0]
    profiles["pc2"] = creator_pcs[:, 1]
    profiles = profiles.groupby("bmu", sort=True).mean()
    profiles.insert(0, "n_creators", np.bincount(bmu, minlength=n_neurons)[profiles.index])
    write_csv(ctx.artifact("creator_profiles.csv"), profiles.reset_index())

    explained_pct = [100.0 * float(r) for r in pca.explained_variance_ratio[:2]]
    if ctx.plots:
        _plot_training_curve(history_repeats, params["n_repeats"], ctx.artifact("training_curve.png"))
        topic_by_name = dict(zip(names, topics.tolist()))
        _plot_hit_map(grid, hits, labels, topic_by_name, cfg.topology, ctx.seed, ctx.artifact("hit_map.png"))
        _plot_u_matrix(grid, u, hits, cfg.topology, ctx.seed, ctx.artifact("u_matrix.png"))
        _plot_component_planes(grid, prototypes, cfg.topology, ctx.seed, ctx.artifact("component_planes.png"))
        _plot_pc_planes(grid, neuron_pcs, explained_pct, cfg.topology, ctx.seed, ctx.artifact("pc_planes.png"))
        _plot_biplot_cells(creator_pcs, bmu, names, grid, explained_pct, ctx.seed, ctx.artifact("biplot_cells.png"))

    n_empty = int((hits == 0).sum())
    aggregates = {}
    for metric in REPEAT_METRICS:
        aggregates[f"{metric}_mean"] = float(per_seed[metric].mean())
        aggregates[f"{metric}_std"] = float(per_seed[metric].std(ddof=1))
    return {
        "input": "pca",
        "pca_config": ctx.config["pca_config"],
        "variance_threshold": params["variance_threshold"],
        "retained_variance": pca_input.retained_variance,
        "n_components": pca_input.n_components,
        "n_original_features": int(Z.shape[1]),
        "projection_rmse": pca_input.projection_rmse,
        "prototype_interpretation": "approximate reconstruction from retained principal components",
        "n_repeats": params["n_repeats"],
        "representative_seed": ctx.seed,
        "seeds": per_seed["seed"].tolist(),
        **aggregates,
        "n_samples": int(Z.shape[0]),
        "n_features": int(scores.shape[1]),
        "rows": grid.rows,
        "cols": grid.cols,
        "n_neurons": n_neurons,
        "topology": cfg.topology,
        "init": cfg.init,
        "epochs": cfg.epochs,
        "iterations": cfg.epochs * int(Z.shape[0]),
        "eta0": round(cfg.eta0, 4),
        "radius0": round(cfg.radius0, 4),
        "radius_decay": cfg.radius_decay,
        "mean_bmu_distance": round(float(result.history["mean_bmu_distance"][-1]), 4),
        "mean_bmu_distance_original": float(bmu_dist_original.mean()),
        "n_empty_neurons": n_empty,
        "empty_neuron_frac": round(n_empty / n_neurons, 4),
        "max_hits": int(hits.max()),
        "mean_hits_occupied": round(int(Z.shape[0]) / (n_neurons - n_empty), 4),
        "n_creators_mapped": int(len(bmu)),
        "u_mean": round(float(np.nanmean(u)), 4),
        "u_max": round(float(np.nanmax(u)), 4),
        "u_max_edge": f"{int(edge_table['neuron_a'].iloc[0])}~{int(edge_table['neuron_b'].iloc[0])}",
        "pc1_explained_variance_pct": round(float(pca.explained_variance_ratio[0]) * 100.0, 1),
        "pc2_explained_variance_pct": round(float(pca.explained_variance_ratio[1]) * 100.0, 1),

    }


EXPERIMENT = Experiment(
    kind="som",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
    sweep_report=report,
)
