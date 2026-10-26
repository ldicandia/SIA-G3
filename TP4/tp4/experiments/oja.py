"""Índice de Oja y comparación con PC1 sobre los mismos datos estandarizados."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.pca import fit_pca, orient_components, standardize
from ..errors import ConfigError
from ..features import dominant_topic, load_design
from ..models.oja import OjaConfig, OjaResult, norm_band_epoch, train_oja
from ..plotting import plt, save_figure
from ..runs import Experiment, RunContext, write_csv
from .pca import TOPIC_COLORS, TOPIC_LABELS

EXTRA_KEYS = frozenset({"data", "feature_set"})
PARAM_DEFAULTS: dict[str, Any] = {
    "epochs": 200,
    "eta0": 0.001,
    "norm_tol": 0.01,
    "n_repeats": 30,
}
MAX_EPOCHS = 5000


def _is_int(v: Any) -> bool:
    return type(v) is int


def _is_real(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def validate(config: dict) -> None:
    for key in ("data", "feature_set"):
        v = config.get(key)
        if not isinstance(v, str) or not v:
            raise ConfigError(f"oja: {key} must be a non-empty string path, got {v!r}")
    p = config["params"]
    if not (_is_int(p["epochs"]) and 1 <= p["epochs"] <= MAX_EPOCHS):
        raise ConfigError(f"oja: params.epochs must be an integer in [1, {MAX_EPOCHS}], got {p['epochs']!r}")
    eta0 = p["eta0"]
    if not (_is_real(eta0) and 0 < eta0 <= 1):
        raise ConfigError(f"oja: params.eta0 must be in (0, 1], got {eta0!r}")
    if not (_is_int(p["n_repeats"]) and 2 <= p["n_repeats"] <= 200):
        raise ConfigError("oja: params.n_repeats must be an integer in [2, 200]")
    tol = p["norm_tol"]
    if not (_is_real(tol) and 0 < tol < 1):
        raise ConfigError(f"oja: params.norm_tol must be a number with 0 < norm_tol < 1, got {tol!r}")


def oja_config_from_params(params) -> OjaConfig:
    return OjaConfig(
        epochs=int(params["epochs"]),
        eta0=float(params["eta0"]),
    )


@dataclass(frozen=True)
class IndexFit:
    result: OjaResult
    w_hat: np.ndarray | None
    index: np.ndarray | None
    flipped: bool | None


def fit_index(Z: np.ndarray, cfg: OjaConfig, rng: np.random.Generator, reference: np.ndarray | None) -> IndexFit:
    """Train the neuron and turn its vector into the oriented creator index (shared with the campaigns kind)."""
    result = train_oja(Z, cfg, rng, reference=reference)
    if result.diverged:
        return IndexFit(result=result, w_hat=None, index=None, flipped=None)
    u = result.weights / np.linalg.norm(result.weights)
    if reference is None:
        oriented, _ = orient_components(u[None, :], (Z @ u)[:, None])
        w_hat = oriented[0]
    else:
        w_hat = u if float(u @ reference) >= 0 else -u
    return IndexFit(result=result, w_hat=w_hat, index=Z @ w_hat, flipped=bool(w_hat @ u < 0))


def rank_order(values) -> np.ndarray:
    """Creator positions from the highest value down; equal values keep their CSV order."""
    return np.argsort(-np.asarray(values, dtype=float), kind="stable")


def _top_feature(coefs: np.ndarray, features: list[str]) -> str:
    return features[int(np.argmax(np.abs(coefs)))]  # first maximum, as orient_components


def _mean_band(ax, table, metric, color, label):
    x = table["epoch"].to_numpy()
    mean = table[f"{metric}_mean"].to_numpy()
    std = table[f"{metric}_std"].to_numpy()
    counts = table[f"{metric}_n"]
    lo, hi = int(counts.min()), int(counts.max())
    n_label = f"n={lo}" if lo == hi else f"n={lo}–{hi}"
    ax.plot(x, mean, color=color, label=f"{label} ({n_label})")
    ax.fill_between(x, mean - std, mean + std, where=table[f"{metric}_n"].to_numpy() >= 2,
                    color=color, alpha=0.22, label="±1 DE muestral entre semillas")


def _plot_convergence(table, norm_tol, path: Path) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    _mean_band(top, table, "w_norm", "#0072B2", "norma media de w")
    top.axhline(1.0, color="#444444", linewidth=1.0)
    top.axhspan(1.0 - norm_tol, 1.0 + norm_tol, color="#999999", alpha=0.15,
                label=f"banda de norma 1 ±{norm_tol:g}")
    top.set_ylabel("‖w‖")
    top.legend(fontsize=8)
    top.grid(alpha=0.3)
    top.set_title("Oja: norma y cambio de pesos entre semillas")
    _mean_band(bottom, table, "delta_w", "#CC79A7", "cambio medio por época")
    bottom.set_ylabel("‖Δw‖")
    bottom.set_xlabel("época")
    bottom.legend(fontsize=8)
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, path)


def _plot_cosine(table, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _mean_band(ax, table, "cos_to_pc1", "#0072B2", "alineación media")
    ax.axhline(1.0, color="#444444", linewidth=1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("época")
    ax.set_ylabel("|cos(w, PC1)|")
    ax.set_title("Oja: alineación con PC1 entre semillas")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_figure(fig, path)


def _plot_weights(table: pd.DataFrame, path: Path) -> None:
    ordered = table.reindex(table["pc1_loading"].abs().sort_values(ascending=False, kind="stable").index)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(ordered) + 2.0))
    ax.barh(y - 0.2, ordered["oja_weight_mean"], xerr=ordered["oja_weight_std"], height=0.4,
            color="#0072B2", capsize=2,
            label=f"Oja: media ±1 DE (n={int(table['n_repeats'].min())} semillas)")
    ax.barh(y + 0.2, ordered["pc1_loading"], height=0.4, color="#E69F00", label="PC1")
    ax.set_yticks(y, labels=list(ordered["feature"]), fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("coeficientes de vectores normalizados")
    ax.set_title("Pesos de Oja entre semillas frente a las cargas de PC1")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    save_figure(fig, path)


def _plot_scores(index, index_std, pc1_scores, names, topics, order, n_repeats, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for topic, color in TOPIC_COLORS.items():
        mask = topics == topic
        if mask.any():
            ax.errorbar(pc1_scores[mask], index[mask], yerr=index_std[mask], fmt="o",
                        markersize=4, color=color, capsize=2, linewidth=0.7,
                        label=TOPIC_LABELS[topic], zorder=3)
    lim = max(float(np.abs(pc1_scores).max()), float(np.abs(index).max())) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#444444", linewidth=1.0, linestyle="--", label="identidad")
    for group, x_label, ha, y_start, y_step in ((order[:5][::-1], 0.78, "left", 0.66, -0.05), (order[-5:][::-1], 0.22, "right", 0.54, -0.05)):
        for j, i in enumerate(group):
            ax.annotate(str(names[i]), (pc1_scores[i], index[i]), xytext=(x_label, y_start + y_step * j),
                        textcoords="axes fraction", ha=ha, va="center", fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": "#888888", "linewidth": 0.6, "shrinkA": 2, "shrinkB": 3})
    ax.set_xlabel("puntaje en PC1")
    ax.set_ylabel("índice de Oja: media ±1 DE entre semillas")
    ax.set_title(f"Índice de Oja frente a PC1 (n={n_repeats} semillas)")
    ax.legend(title="tópico dominante", fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    save_figure(fig, path)


def _repeated_analysis(ctx, Z, cfg, pc1, pc1_scores, ids, features, first):
    seeds = list(range(ctx.seed, ctx.seed + ctx.params["n_repeats"]))
    fits = [first]
    fits.extend(fit_index(Z, cfg, np.random.default_rng(seed), pc1) for seed in seeds[1:])
    rows, histories, weights, indices = [], [], [], []
    for seed, fit in zip(seeds, fits, strict=True):
        result = fit.result
        history = pd.DataFrame(result.history).rename(columns={"cos_to_reference": "cos_to_pc1"})
        history.insert(0, "seed", seed)
        histories.append(history)
        record = {"seed": seed, "diverged": result.diverged,
                  "divergence_epoch": result.divergence_epoch,
                  "final_w_norm": None, "final_delta_w": None,
                  "cosine_to_pc1": None, "pearson_scores_pc1": None,
                  "norm_band_epoch": norm_band_epoch(result.history["w_norm"], ctx.params["norm_tol"])}
        if not result.diverged:
            record.update(final_w_norm=float(np.linalg.norm(result.weights)),
                          final_delta_w=float(result.history["delta_w"][-1]),
                          cosine_to_pc1=min(1.0, abs(float(fit.w_hat @ pc1))),
                          pearson_scores_pc1=float(np.corrcoef(fit.index, pc1_scores)[0, 1]))
            weights.append(pd.DataFrame({"seed": seed, "feature": features, "oja_weight": fit.w_hat}))
            order = rank_order(fit.index)
            ranks = np.empty(len(order), dtype=int)
            ranks[order] = np.arange(1, len(order) + 1)
            frame = ids.copy()
            frame.insert(0, "seed", seed)
            frame["oja_index"] = fit.index
            frame["rank"] = ranks
            indices.append(frame)
        rows.append(record)
    repeats = pd.DataFrame(rows)
    write_csv(ctx.artifact("repeats.csv"), repeats)
    history_all = pd.concat(histories, ignore_index=True)
    write_csv(ctx.artifact("history_repeats.csv"), history_all)
    metrics = ["w_norm", "delta_w", "cos_to_pc1"]
    stats = history_all.groupby("epoch", sort=True)[metrics].agg(["mean", "std", "count"])
    stats.columns = [f"{metric}_{'n' if stat == 'count' else stat}" for metric, stat in stats.columns]
    stats = stats.reset_index()
    write_csv(ctx.artifact("history_stats.csv"), stats)
    summary = {"n_repeats": len(seeds), "repeat_seeds": seeds,
               "n_successful_repeats": int((~repeats["diverged"]).sum()),
               "n_diverged_repeats": int(repeats["diverged"].sum()),
               "dispersion": "sample standard deviation across training seeds"}
    for metric in ("final_w_norm", "final_delta_w", "cosine_to_pc1", "pearson_scores_pc1", "norm_band_epoch"):
        values = repeats[metric].dropna().astype(float)
        summary[f"{metric}_mean"] = float(values.mean()) if len(values) else None
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) >= 2 else None
        summary[f"{metric}_n"] = len(values)
    weight_stats = index_stats = None
    if weights:
        weight_all = pd.concat(weights, ignore_index=True)
        write_csv(ctx.artifact("weights_repeats.csv"), weight_all)
        weight_stats = (weight_all.groupby("feature", sort=False)["oja_weight"]
                        .agg(oja_weight_mean="mean", oja_weight_std="std", n_repeats="count")
                        .reindex(features).reset_index())
        weight_stats["pc1_loading"] = pc1
        write_csv(ctx.artifact("weights_stats.csv"), weight_stats)
        index_all = pd.concat(indices, ignore_index=True)
        write_csv(ctx.artifact("index_repeats.csv"), index_all)
        index_stats = (index_all.groupby("creator_id", sort=False)["oja_index"]
                       .agg(oja_index_mean="mean", oja_index_std="std", n_repeats="count")
                       .reindex(ids["creator_id"]).reset_index())
        index_stats.insert(1, "name", ids["name"].to_numpy())
        index_stats["pc1_score"] = pc1_scores
        top5 = index_all.loc[index_all["rank"] <= 5, "creator_id"].value_counts()
        index_stats["top5_count"] = index_stats["creator_id"].map(top5).fillna(0).astype(int)
        write_csv(ctx.artifact("index_stats.csv"), index_stats)
    if ctx.plots:
        _plot_convergence(stats, ctx.params["norm_tol"], ctx.artifact("convergence.png"))
        _plot_cosine(stats, ctx.artifact("cos_to_pc1.png"))
        if weight_stats is not None:
            _plot_weights(weight_stats, ctx.artifact("weights_vs_pc1.png"))
    return summary, weight_stats, index_stats


def run(ctx: RunContext) -> dict:
    params = ctx.params
    cfg = oja_config_from_params(params)
    norm_tol = float(params["norm_tol"])
    design = load_design(ctx.config["data"], ctx.config["feature_set"])
    Z = standardize(design.X)[0]
    features = list(design.X.columns)
    pca = fit_pca(Z, features)
    pc1 = pca.components[0]

    fit = fit_index(Z, cfg, ctx.rng, pc1)
    result = fit.result
    ids = design.ids.reset_index(drop=True)
    repeated_summary, _, index_stats = _repeated_analysis(
        ctx, Z, cfg, pc1, pca.scores[:, 0], ids, features, fit
    )

    history = pd.DataFrame(result.history).rename(columns={"cos_to_reference": "cos_to_pc1"})
    write_csv(ctx.artifact("history.csv"), history)
    norms = result.history["w_norm"]
    converge = norm_band_epoch(norms, norm_tol)

    summary: dict[str, Any] = {
        "input": "features",
        "n_samples": int(Z.shape[0]),
        "n_features": int(Z.shape[1]),
        "epochs": cfg.epochs,
        "eta0": cfg.eta0,
        "norm_tol": norm_tol,
        "init": "uniform01",
        "diverged": bool(result.diverged),
        "divergence_epoch": result.divergence_epoch,
        "initial_w_norm": round(float(np.linalg.norm(result.initial_weights)), 4),
        **repeated_summary,
    }
    if result.diverged:
        summary.update(
            {
                "final_w_norm": None,
                "final_delta_w": None,
                "norm_band_epoch": None,
                "cosine_to_pc1": None,
                "pearson_scores_pc1": None,
                "orientation_matches_pc1": None,
                "orientation_flipped": None,
                "top_creator": None,
                "bottom_creator": None,
            }
        )
        return summary

    index, w_hat = fit.index, fit.w_hat
    order = rank_order(index)
    topics = dominant_topic(design.raw)
    ids = design.ids.reset_index(drop=True)
    names_all = ids["name"].to_numpy()
    raw = design.raw

    ranking = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "creator_id": ids["creator_id"].to_numpy()[order],
            "name": names_all[order],
            "dominant_topic": topics.to_numpy()[order],
            "topic_football_pct": raw["topic_football_pct"].to_numpy()[order],
            "topic_makeup_pct": raw["topic_makeup_pct"].to_numpy()[order],
            "topic_dogs_pct": raw["topic_dogs_pct"].to_numpy()[order],
            "topic_art_pct": raw["topic_art_pct"].to_numpy()[order],
            "followers": raw["followers"].to_numpy()[order],
            "oja_index": index[order],
            "pc1_score": pca.scores[order, 0],
        }
    )
    write_csv(ctx.artifact("ranking.csv"), ranking)

    weights = pd.DataFrame(
        {
            "feature": features,
            "oja_weight": w_hat,
            "pc1_loading": pc1,
            "difference": w_hat - pc1,
            "oja_weight_raw": result.weights,
        }
    )
    write_csv(ctx.artifact("weights_vs_pc1.csv"), weights)

    if ctx.plots and index_stats is not None:
        mean_index = index_stats["oja_index_mean"].to_numpy()
        _plot_scores(mean_index, index_stats["oja_index_std"].to_numpy(), pca.scores[:, 0],
                     names_all, topics.to_numpy(), rank_order(mean_index), int(index_stats["n_repeats"].min()),
                     ctx.artifact("scores_vs_pc1.png"))

    names = ranking["name"].tolist()
    topic_of = ranking["dominant_topic"].tolist()
    lambda1 = float(pca.eigenvalues[0])
    summary.update(
        {
            "final_w_norm": round(float(norms[-1]), 4),
            "final_delta_w": round(float(result.history["delta_w"][-1]), 6),
            "norm_band_epoch": converge,
            "cosine_to_pc1": round(min(1.0, abs(float(w_hat @ pc1))), 6),
            "pearson_scores_pc1": round(float(np.corrcoef(index, pca.scores[:, 0])[0, 1]), 6),
            "orientation_matches_pc1": bool(w_hat @ pc1 > 0),
            "orientation_flipped": fit.flipped,
            "oja_top_feature": _top_feature(w_hat, features),
            "pc1_top_feature": _top_feature(pc1, features),
            "max_abs_weight_diff": round(float(np.max(np.abs(w_hat - pc1))), 6),
            "lambda1": round(lambda1, 4),
            "course_eta_bound": round(1.0 / (1.2 * lambda1), 4),
            "top_creator": names[0],
            "top_creator_topic": topic_of[0],
            "bottom_creator": names[-1],
            "bottom_creator_topic": topic_of[-1],
            "top5_creators": names[:5],
            "bottom5_creators": names[-5:],
        }
    )
    return summary


EXPERIMENT = Experiment(
    kind="oja",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
