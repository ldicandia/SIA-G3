"""Experiment kind `som_control`: the Kohonen map on the 25 features against a map on the first PCs.

Section 1.1 of docs/hallazgos/ej1.md decided to train Kohonen on the standardized features and
promised a control map on the first principal components (k_cum_80 = 6). This kind trains both
maps with the base config's hyperparameters and the same seed and compares them.

Config keys (besides the common ones):
  base                path to the `som` config whose hyperparameters and data are used (relative to TP4/)
  params.pca_components   number of leading principal components for the control map (default 6)

The control seed must equal the base seed: with it, the features map inside this run is the
base run's map exactly (tests assert equal QE, TE and distance preservation).

Artifacts:
  control_comparison.csv   QE, TE, distance preservation, empty fraction, n_regions for both maps
  creator_regions.csv      each creator's winning neuron and region on both maps
  region_contingency.csv   creators per (region on the features map, region on the control map)
  control_region_maps.png  both region maps side by side

`tp4.runner` is imported only inside functions: the runner imports the experiment registry,
which imports this module (a top-level import would cycle).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.pca import fit_pca, standardize
from ..analysis.som_compare import adjusted_rand_index, contingency_table, grid_distance_agreement
from ..analysis.som_maps import REGION_K_RANGE, choose_n_regions, region_scan, segment_regions
from ..config import load_config
from ..errors import ConfigError
from ..features import dominant_topic, load_design
from ..models.som import best_two_units, distance_preservation, quantization_error, topographic_error, train_som
from ..plotting import draw_grid_cells, draw_region_borders, plt, save_figure
from ..runs import Experiment, RunContext, resolve_project_path, write_csv
from .pca import TOPIC_COLORS, TOPIC_LABELS
from .som import _border_pairs, _draw_names, labels_by_neuron, som_config_from_params

EXTRA_KEYS = frozenset({"base"})
PARAM_DEFAULTS: dict[str, Any] = {"pca_components": 6}


def _load_base(config) -> dict[str, Any]:
    """Load and validate the base config; it must be a `som` run (read only through load_config)."""
    from .. import runner

    base = config.get("base")
    if not isinstance(base, str) or not base:
        raise ConfigError(f"som_control: base must be a non-empty string path, got {base!r}")
    loaded = load_config(resolve_project_path(base))
    if loaded.get("kind") != "som":
        raise ConfigError(f"som_control: base must be a config of kind 'som', got kind {loaded.get('kind')!r}")
    return runner.validate_config(loaded)


def validate(config: dict) -> None:
    base = _load_base(config)
    if config["seed"] != base["seed"]:
        raise ConfigError(
            f"som_control: seed {config['seed']!r} must equal the base seed {base['seed']!r} "
            "so the features map reproduces the base run"
        )
    k = config["params"]["pca_components"]
    if type(k) is not int or k < 1:
        raise ConfigError(f"som_control: params.pca_components must be an integer >= 1, got {k!r}")


def _num(value: float, digits: int = 4) -> float | None:
    value = float(value)
    return None if math.isnan(value) else round(value, digits)


def _plot_control_maps(grid, topology, panels, topic_by_name, path) -> None:
    height_unit = math.sqrt(3) / 2 if topology == "hex" else 1.0
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * (1.5 * grid.cols + 1.2), 1.5 * grid.rows * height_unit + 2.6)
    )
    cmap = plt.get_cmap("Pastel1")
    for ax, (title, region_labels, counts, labels) in zip(axes, panels):
        colors = [cmap(r % cmap.N) for r in range(len(counts))]
        draw_grid_cells(ax, grid.positions, topology, colors=[colors[int(r) - 1] for r in region_labels])
        _draw_names(ax, grid, labels, topic_by_name)
        draw_region_borders(ax, grid.positions, topology, _border_pairs(grid, region_labels))
        ax.set_title(title, fontsize=10)
        handles = [
            plt.Line2D([], [], marker="s", markersize=8, linestyle="", color=colors[r], markeredgecolor="gray",
                       label=f"región {r + 1} ({int(counts[r])})")
            for r in range(len(counts))
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=min(len(counts), 3),
                  frameon=False, fontsize=8)
    topic_handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=TOPIC_COLORS[t], label=TOPIC_LABELS[t])
        for t in TOPIC_COLORS
    ]
    fig.legend(handles=topic_handles, title="tópico dominante", loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Mapa sobre las variables vs. mapa de control sobre componentes principales (mismos hiperparámetros y semilla)")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    save_figure(fig, path)


def run(ctx: RunContext) -> dict:
    base = _load_base(ctx.config)
    if ctx.config["seed"] != base["seed"]:  # validate() ran already; kept so run() is safe on its own
        raise ConfigError(f"som_control: seed {ctx.config['seed']!r} must equal the base seed {base['seed']!r}")
    n_pcs = int(ctx.params["pca_components"])

    design = load_design(base["data"], base["feature_set"])
    Z = standardize(design.X)[0]
    n_features = Z.shape[1]
    if n_pcs > n_features:
        raise ConfigError(
            f"som_control: params.pca_components is {n_pcs} but the feature set has only {n_features} features"
        )
    feature_names = list(design.X.columns)
    pca = fit_pca(Z, feature_names, n_components=n_pcs)
    S = pca.scores
    cfg = som_config_from_params(base["params"])
    seed = int(base["seed"])

    map_f = train_som(Z, cfg, np.random.default_rng(seed), xp=ctx.backend.xp)
    map_p = train_som(S, cfg, np.random.default_rng(seed), xp=ctx.backend.xp)
    grid = map_f.grid
    n_neurons = grid.n_neurons

    bmu_f = best_two_units(Z, map_f.weights)[0]
    bmu_p = best_two_units(S, map_p.weights)[0]
    metrics = {}
    for tag, X, result, bmu in (("features", Z, map_f, bmu_f), ("pca", S, map_p, bmu_p)):
        hits = np.bincount(bmu, minlength=n_neurons)
        metrics[tag] = {
            "quantization_error": quantization_error(X, result.weights),
            "topographic_error": topographic_error(X, result.weights, grid),
            "distance_preservation": distance_preservation(X, result.weights, grid),
            "empty_neuron_frac": float((hits == 0).sum()) / n_neurons,
        }

    # Same k on both maps: the base's explicit n_regions, or the recorded rule applied to the features map.
    k_values = list(range(REGION_K_RANGE[0], min(REGION_K_RANGE[1], n_neurons - 1) + 1))
    base_k = base["params"]["n_regions"]
    k = int(base_k) if base_k is not None else choose_n_regions(region_scan(map_f.weights, grid, bmu_f, k_values))
    regions_f = segment_regions(map_f.weights, grid, k)
    regions_p = segment_regions(map_p.weights, grid, k)
    creator_f = regions_f[bmu_f]
    creator_p = regions_p[bmu_p]

    ari = adjusted_rand_index(creator_f, creator_p)
    agreement = grid_distance_agreement(grid.distances, bmu_f, grid.distances, bmu_p)

    comparison = pd.DataFrame(
        {
            "metric": [*metrics["features"], "n_regions"],
            "features_map": [*metrics["features"].values(), k],
            "pca_map": [*metrics["pca"].values(), k],
        }
    )
    write_csv(ctx.artifact("control_comparison.csv"), comparison)

    topics = dominant_topic(design.raw)
    names = design.raw["name"].tolist()
    ids = design.ids.reset_index(drop=True)
    creator_table = pd.DataFrame(
        {
            "creator_id": ids["creator_id"],
            "name": ids["name"],
            "dominant_topic": topics.reset_index(drop=True),
            "bmu_features": bmu_f,
            "region_features": creator_f,
            "bmu_pca": bmu_p,
            "region_pca": creator_p,
        }
    )
    write_csv(ctx.artifact("creator_regions.csv"), creator_table)

    contingency = contingency_table(creator_f, creator_p)
    contingency.index.name = "region_features"
    contingency.columns = [f"pca_region_{c}" for c in contingency.columns]
    write_csv(ctx.artifact("region_contingency.csv"), contingency.reset_index())

    kept_pct = round(float(pca.cumulative_ratio[n_pcs - 1]) * 100.0, 1)
    if ctx.plots:
        topic_by_name = dict(zip(names, topics.tolist()))
        panels = []
        for title, region_labels, bmu in (
            ("Mapa sobre las 25 variables estandarizadas", regions_f, bmu_f),
            (f"Mapa de control sobre {n_pcs} componentes principales ({str(kept_pct).replace('.', ',')}% de la varianza)", regions_p, bmu_p),
        ):
            counts = np.bincount(region_labels[bmu], minlength=k + 1)[1:]
            panels.append((title, region_labels, counts, labels_by_neuron(bmu, names, n_neurons)))
        _plot_control_maps(grid, cfg.topology, panels, topic_by_name, ctx.artifact("control_region_maps.png"))

    return {
        "base_experiment": base["experiment"],
        "pca_components": n_pcs,
        "pca_variance_kept_pct": kept_pct,
        "n_regions": k,
        "features_quantization_error": _num(metrics["features"]["quantization_error"]),
        "pca_quantization_error": _num(metrics["pca"]["quantization_error"]),
        "features_topographic_error": _num(metrics["features"]["topographic_error"]),
        "pca_topographic_error": _num(metrics["pca"]["topographic_error"]),
        "features_distance_preservation": _num(metrics["features"]["distance_preservation"]),
        "pca_distance_preservation": _num(metrics["pca"]["distance_preservation"]),
        "features_empty_neuron_frac": _num(metrics["features"]["empty_neuron_frac"]),
        "pca_empty_neuron_frac": _num(metrics["pca"]["empty_neuron_frac"]),
        "region_ari": _num(ari),
        "bmu_distance_agreement": _num(agreement),
    }


EXPERIMENT = Experiment(
    kind="som_control",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
