"""Assignment 1.4: join PCA/Oja/SOM evidence and compare explicit campaign assumptions."""

import numpy as np
import pandas as pd

from ..analysis.campaigns import (RANKABLE_COLUMNS, build_evidence, describe_rule,
                                  followers_top, pick, cell_profiles, strategy_metrics)
from ..analysis.pca import fit_pca, standardize
from ..config import load_config
from ..errors import ConfigError
from ..features import TOPIC_COLUMNS, load_design
from ..plotting import plt, save_figure
from ..runs import Experiment, resolve_project_path, write_csv
from .oja import fit_index, oja_config_from_params
from .som import load_som_config, som_evidence

EXTRA_KEYS = frozenset({"oja_config", "som_config", "clients"})
PARAM_DEFAULTS = {"n_picks": 3}


def _load_oja_base(config):
    from ..runner import validate_config
    path = config.get("oja_config")
    if not isinstance(path, str) or not path:
        raise ConfigError("campaigns: oja_config is required")
    base = load_config(resolve_project_path(path))
    if base.get("kind") != "oja":
        raise ConfigError("campaigns: oja_config must have kind oja")
    return validate_config(base)


def validate(config):
    base = _load_oja_base(config)
    som_path = config.get("som_config")
    if not isinstance(som_path, str) or not som_path:
        raise ConfigError("campaigns: som_config is required")
    som = load_som_config(som_path, who="campaigns")
    if any(base[key] != som[key] for key in ("data", "feature_set")):
        raise ConfigError("campaigns: Oja and SOM must use the same data and feature_set")
    n = config["params"]["n_picks"]
    if type(n) is not int or not 1 <= n <= 10:
        raise ConfigError("campaigns: n_picks must be an integer in [1, 10]")
    clients = config.get("clients")
    if not isinstance(clients, list) or not 1 <= len(clients) <= 8:
        raise ConfigError("campaigns: provide 1 to 8 clients")
    names = set()
    required = {"name", "label", "objective", "topic", "min_share", "order_by"}
    for client in clients:
        if not isinstance(client, dict) or set(client) != required:
            raise ConfigError(f"campaigns: client keys must be {sorted(required)}")
        for key in ("name", "label", "objective"):
            if not isinstance(client[key], str) or not client[key].strip():
                raise ConfigError(f"campaigns: {key} must be a nonempty string")
        if client["name"] in names:
            raise ConfigError("campaigns: duplicate client name")
        names.add(client["name"])
        share = client["min_share"]
        if type(share) not in (int, float) or not np.isfinite(share) or not 0 <= share <= 100:
            raise ConfigError("campaigns: min_share must be in [0, 100]")
        if client["topic"] not in TOPIC_COLUMNS or client["order_by"] not in RANKABLE_COLUMNS:
            raise ConfigError("campaigns: invalid topic or order_by")


def _plot_comparison(table, clients, path):
    fig, axes = plt.subplots(len(clients), 2, figsize=(10, 2.5 * len(clients)), squeeze=False)
    columns = [("mean_topic_share", "Contenido del tópico (%)"),
               ("mean_saves_per_1000", "Guardados por 1000")]
    for i, client in enumerate(clients):
        rows = table.loc[table["client"] == client["name"]]
        for j, (column, label) in enumerate(columns):
            axes[i, j].bar(rows["strategy"], rows[column], color=["#332288", "#999999", "#D55E00"])
            axes[i, j].set_title(f"{client['label']}: {label}", fontsize=9)
            axes[i, j].tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    save_figure(fig, path)


def run(ctx):
    base = _load_oja_base(ctx.config)
    som_base = load_som_config(ctx.config["som_config"], who="campaigns")
    design = load_design(base["data"], base["feature_set"])
    Z = standardize(design.X)[0]
    pca = fit_pca(Z, list(design.X.columns))
    oja = fit_index(Z, oja_config_from_params(base["params"]),
                    np.random.default_rng(base["seed"]), pca.components[0])
    if oja.index is None:
        raise ConfigError("campaigns: Oja diverged")
    som = som_evidence(som_base, Z)
    evidence = build_evidence(design.raw, pca.scores[:, 0], pca.scores[:, 1], oja.index,
                              som.bmu, som.row, som.col)
    write_csv(ctx.artifact("evidence.csv"), evidence)
    n = ctx.params["n_picks"]
    clients = ctx.config["clients"]
    picks, baselines, comparisons, profiles = [], [], [], []
    summary = {"n_creators": len(evidence), "n_clients": len(clients), "n_picks": n,
               "decision_type": "explicit commercial assumptions; no learned ranking"}
    for client in clients:
        try:
            selected = pick(evidence, client, n)
            global_top = followers_top(evidence, n)
            topic_top = followers_top(evidence, n, evidence[client["topic"]] >= client["min_share"])
        except ValueError as exc:
            raise ConfigError(f"campaigns: {client['name']}: {exc}") from exc
        for strategy, rows in [("assumption", selected), ("followers_global", global_top),
                               ("followers_in_topic", topic_top)]:
            view = rows.copy()
            view.insert(0, "client", client["name"])
            view.insert(1, "strategy", strategy)
            view.insert(2, "pick_rank", np.arange(1, len(view) + 1))
            (picks if strategy == "assumption" else baselines).append(view)
            comparisons.append({"client": client["name"], "strategy": strategy,
                                "creators": ";".join(rows["name"]),
                                **strategy_metrics(rows, client["topic"])})
        profiles.append(cell_profiles(evidence, client))
        summary[f"{client['name']}_picks"] = selected["name"].tolist()
        summary[f"{client['name']}_rule"] = describe_rule(client)
    write_csv(ctx.artifact("picks.csv"), pd.concat(picks, ignore_index=True))
    write_csv(ctx.artifact("baselines.csv"), pd.concat(baselines, ignore_index=True))
    table = pd.DataFrame(comparisons)
    write_csv(ctx.artifact("comparison.csv"), table)
    write_csv(ctx.artifact("strategy_profiles.csv"), pd.concat(profiles, ignore_index=True))
    if ctx.plots:
        _plot_comparison(table, clients, ctx.artifact("comparison.png"))
    summary["followers_global_picks"] = global_top["name"].tolist()
    return summary


EXPERIMENT = Experiment(kind="campaigns", run=run, extra_keys=EXTRA_KEYS,
                        param_defaults=PARAM_DEFAULTS, validate=validate)
