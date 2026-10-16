"""Sweep report for kind `oja`: tabulates, plots and selects the learning rate eta0 (D-04).

`report(ctx, table)` is registered as `Experiment.sweep_report` of the `oja` kind; the generic
sweep calls it after writing sweep.csv. The sweep crosses `params.eta0` with the top-level `seed`
(three seeds), and the report writes:

  selection.csv            one row per eta0: the rule's stages and the numbers behind them
  selected_params.json     {"child": ..., "params": {"params.eta0": value}} with the grid's JSON type
  sweep_convergence.png    ||w|| per epoch, one panel per eta0, one line per seed
  sign_ambiguity.png       at the selected eta0: raw weights per seed against the oriented weights

The rule is PC1-free (prohibition OJA-02): PC1 never selects, it is only reported
(min_cosine_to_pc1). An eta0 is kept when
  stage_stable  no seed diverged,
  stage_norm    every seed ends inside the ||w|| = 1 band (epochs_to_converge is not null),
  stage_index   the creator rankings of all seeds are identical (max rank displacement 0),
and among the kept ones the smallest max-over-seeds epochs_to_converge wins, ties to the smaller eta0.

This module must not import tp4.experiments.oja or the registry (the oja module imports it).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_config
from ..errors import ConfigError
from ..plotting import grid_shape, plt, save_figure
from ..runs import RunContext, resolve_project_path, write_csv, write_json

RANK_DISPLACEMENT_MAX = 0
ETA_KEY = "params.eta0"
SELECTION_COLUMNS = (
    "eta0", "n_seeds", "n_diverged", "max_epochs_to_converge", "max_norm_deviation",
    "max_rank_displacement", "min_cosine_to_pc1", "stage_stable", "stage_norm", "stage_index", "selected",
)
SEED_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def _py(value: Any) -> Any:
    """numpy scalar / NaN to a plain JSON-safe Python value (NaN becomes None)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _truthy(value: Any) -> bool:
    """True only for a real True (None, NaN and pd.NA count as False)."""
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return bool(value)


def max_rank_displacement(rankings: Sequence[pd.Series]) -> int:
    """Max over creators of (max rank - min rank) across the rankings (Series creator_id -> rank)."""
    if len(rankings) < 2:
        return 0
    frame = pd.concat([pd.Series(r) for r in rankings], axis=1)
    return int((frame.max(axis=1) - frame.min(axis=1)).max())


@dataclass(frozen=True)
class EtaSelection:
    table: pd.DataFrame  # one row per eta0 in first-appearance order, SELECTION_COLUMNS
    selected_eta0: float | None


def _nan_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else float("nan")


def _nan_min(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.min()) if finite.size else float("nan")


def select_eta(table: pd.DataFrame, rankings: Mapping[str, pd.Series]) -> EtaSelection:
    """Apply the rule. `table` is sweep.csv; `rankings` maps child name to its rank Series (non-diverged only).

    Pure: no I/O, and the cosine to PC1 is reported but never read by a stage.
    """
    etas = list(dict.fromkeys(float(v) for v in table[ETA_KEY]))
    eta_col = table[ETA_KEY].astype(float).to_numpy()
    diverged = table["diverged"].fillna(False).astype(bool).to_numpy()
    epochs = pd.to_numeric(table["epochs_to_converge"], errors="coerce").to_numpy(dtype=float)
    norms = pd.to_numeric(table["final_w_norm"], errors="coerce").to_numpy(dtype=float)
    cosines = pd.to_numeric(table["cosine_to_pc1"], errors="coerce").to_numpy(dtype=float)
    children = table["child"].astype(str).to_numpy()

    rows = []
    for eta in etas:
        mask = eta_col == eta
        ranks = [rankings[c] for c in children[mask] if c in rankings]
        displacement = float(max_rank_displacement(ranks)) if ranks else float("nan")
        all_converge = bool(np.isfinite(epochs[mask]).all())
        rows.append(
            {
                "eta0": eta,
                "n_seeds": int(mask.sum()),
                "n_diverged": int(diverged[mask].sum()),
                "max_epochs_to_converge": float(epochs[mask].max()) if all_converge else float("nan"),
                "max_norm_deviation": _nan_max(np.abs(norms[mask] - 1.0)),
                "max_rank_displacement": displacement,
                "min_cosine_to_pc1": _nan_min(cosines[mask]),
                "stage_stable": not bool(diverged[mask].any()),
                "stage_norm": all_converge,
                "stage_index": bool(displacement <= RANK_DISPLACEMENT_MAX),  # NaN compares False
                "selected": False,
            }
        )
    out = pd.DataFrame(rows, columns=list(SELECTION_COLUMNS))
    passing = out[out["stage_stable"] & out["stage_norm"] & out["stage_index"]]
    selected: float | None = None
    if len(passing):
        best = passing.sort_values(["max_epochs_to_converge", "eta0"], kind="stable").iloc[0]
        selected = float(best["eta0"])
        out["selected"] = out["eta0"] == selected
    return EtaSelection(table=out, selected_eta0=selected)


def rule_description() -> str:
    return (
        "Se descartan los eta0 con alguna semilla divergente; luego los eta0 en que alguna semilla no termina "
        "con ‖w‖ dentro de la banda alrededor de 1 (norm_tol); luego los eta0 cuyas semillas ordenan a los "
        "creadores distinto (desplazamiento máximo de ranking mayor que 0); entre los que quedan se elige el que "
        "converge en menos épocas (el máximo entre semillas) y, en empate, el eta0 menor. PC1 no interviene en la selección."
    )


# --------------------------------------------------------------------------- plots


def _seed_color(position: int) -> str:
    return SEED_COLORS[position % len(SEED_COLORS)]


def _plot_sweep_convergence(table, etas, seeds, selected_eta, norm_tol, run_dir: Path, path: Path) -> None:
    nrows, ncols = grid_shape(len(etas), 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows), squeeze=False, sharey=True)
    flat = list(axes.ravel())
    eta_col = table[ETA_KEY].astype(float).to_numpy()
    for ax, eta in zip(flat, etas):
        sub = table[eta_col == eta]
        for position, seed in enumerate(seeds):
            match = sub[sub["seed"] == seed] if "seed" in sub else sub
            if match.empty:
                continue
            child = str(match["child"].iloc[0])
            history = pd.read_csv(run_dir / child / "history.csv")
            ax.plot(history["epoch"], history["w_norm"], color=_seed_color(position), linewidth=1.3, label=f"semilla {seed}")
            if bool(match["diverged"].iloc[0]):
                k = int(match["divergence_epoch"].iloc[0])
                ax.text(
                    0.97, 0.9 - 0.12 * position, f"semilla {seed}: diverge en la época {k}", transform=ax.transAxes,
                    ha="right", fontsize=7, color=_seed_color(position),
                )
        ax.axhline(1.0, color="#444444", linewidth=0.9)
        ax.axhspan(1.0 - norm_tol, 1.0 + norm_tol, color="#009E73", alpha=0.18)
        ax.set_ylim(0, 4)
        ax.grid(alpha=0.3)
        is_selected = selected_eta is not None and eta == selected_eta
        ax.set_title(f"η0 = {eta:g}" + ("  (elegido)" if is_selected else ""), fontsize=10, fontweight="bold" if is_selected else "normal")
        if is_selected:
            for spine in ax.spines.values():
                spine.set_edgecolor("#D55E00")
                spine.set_linewidth(2.4)
    for ax in flat[len(etas):]:
        ax.axis("off")
    flat[0].legend(loc="upper right", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("época")
    for row in axes:
        row[0].set_ylabel("‖w‖ (recortado a [0, 4])")
    fig.suptitle("Barrido de η0: norma de w por época, una línea por semilla (banda verde = norm_tol)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path)


def _plot_sign_ambiguity(weights_by_seed: dict[Any, pd.DataFrame], eta: float, path: Path) -> None:
    first = next(iter(weights_by_seed.values()))
    features = list(first["feature"])
    x = np.arange(len(features))
    fig, (raw_ax, oriented_ax) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, column, title in (
        (raw_ax, "oja_weight_raw", "Pesos tal como los entrena la regla (antes de orientar)"),
        (oriented_ax, "oja_weight", "Pesos orientados (coeficiente mayor en valor absoluto positivo)"),
    ):
        ax.plot(x, first["pc1_loading"], color="#444444", linewidth=2.2, alpha=0.5, label="cargas de PC1")
        for position, (seed, frame) in enumerate(weights_by_seed.items()):
            ax.plot(x, frame[column], marker="o", markersize=4, linewidth=0.9, color=_seed_color(position), label=f"semilla {seed}")
        ax.axhline(0, color="lightgray", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("coeficiente")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    oriented_ax.set_xticks(x, labels=features, rotation=90, fontsize=8)
    fig.suptitle(f"Ambigüedad de signo con η0 = {eta:g}: la semilla fija el signo que converge, la orientación lo unifica", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path)


# --------------------------------------------------------------------------- hook


def report(ctx: RunContext, table: pd.DataFrame) -> dict:
    grid: dict[str, list] = dict(ctx.config["grid"])
    if ETA_KEY not in grid:
        raise ConfigError(f"oja sweep: the grid must sweep {ETA_KEY!r}, got keys {list(grid)}")
    base_seed = load_config(resolve_project_path(ctx.config["base"]))["seed"]
    has_seed = "seed" in table.columns
    seeds = [int(s) for s in dict.fromkeys(table["seed"])] if has_seed else [int(ctx.seed)]

    # Each non-diverged child's ranking, read from the artifacts the child itself wrote (child names come from expand_grid).
    diverged = table["diverged"].fillna(False).astype(bool).to_numpy()
    rankings: dict[str, pd.Series] = {}
    for child, is_diverged in zip(table["child"].astype(str), diverged):
        if is_diverged:
            continue
        frame = pd.read_csv(ctx.run_dir / child / "ranking.csv", usecols=["creator_id", "rank"])
        rankings[child] = frame.set_index("creator_id")["rank"]

    sel = select_eta(table, rankings)
    write_csv(ctx.artifact("selection.csv"), sel.table)

    etas = sel.table["eta0"].tolist()
    selected_eta = sel.selected_eta0
    selected_child: str | None = None
    chosen: dict[str, Any] = {}
    seeds_flipped: list[int] = []
    seeds_not_flipped: list[int] = []
    if selected_eta is not None:
        # The value comes from the grid list itself so its JSON type survives.
        chosen = {ETA_KEY: next(v for v in grid[ETA_KEY] if float(v) == selected_eta)}
        at_eta = table[table[ETA_KEY].astype(float) == selected_eta]
        preferred = at_eta[at_eta["seed"] == base_seed] if has_seed else at_eta
        selected_child = str((preferred if len(preferred) else at_eta)["child"].iloc[0])
        for _, row in at_eta.iterrows():
            seed = int(row["seed"]) if has_seed else int(ctx.seed)
            (seeds_flipped if _truthy(row["orientation_flipped"]) else seeds_not_flipped).append(seed)
    write_json(ctx.artifact("selected_params.json"), {"child": selected_child, "params": chosen})

    norm_tol = float(pd.to_numeric(table["norm_tol"], errors="coerce").dropna().iloc[0]) if "norm_tol" in table else 0.01
    if ctx.plots:
        _plot_sweep_convergence(table, etas, seeds, selected_eta, norm_tol, ctx.run_dir, ctx.artifact("sweep_convergence.png"))
        if selected_eta is not None:
            at_eta = table[table[ETA_KEY].astype(float) == selected_eta]
            weights = {}
            for _, row in at_eta.iterrows():
                seed = int(row["seed"]) if has_seed else int(ctx.seed)
                weights[seed] = pd.read_csv(ctx.run_dir / str(row["child"]) / "weights_vs_pc1.csv")
            _plot_sign_ambiguity(weights, selected_eta, ctx.artifact("sign_ambiguity.png"))

    selected_row = sel.table[sel.table["selected"]]
    return {
        "selected_child": selected_child,
        "selected_eta0": _py(chosen[ETA_KEY]) if chosen else None,
        "selected_epochs_to_converge": int(selected_row["max_epochs_to_converge"].iloc[0]) if len(selected_row) else None,
        "selection_rule": rule_description(),
        "n_eta": int(len(sel.table)),
        "n_stable": int(sel.table["stage_stable"].sum()),
        "n_norm_ok": int(sel.table["stage_norm"].sum()),
        "n_index_ok": int(sel.table["stage_index"].sum()),
        "seeds": seeds,
        "seeds_flipped": seeds_flipped,
        "seeds_not_flipped": seeds_not_flipped,
    }
