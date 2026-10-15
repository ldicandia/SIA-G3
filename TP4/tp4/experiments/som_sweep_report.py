"""Sweep report for kind `som`: tabulates, plots and selects the Kohonen configuration (SOM-02).

`report(ctx, table)` is registered as `Experiment.sweep_report` of the `som` kind; the generic
sweep calls it after writing sweep.csv. It applies `tp4.analysis.som_selection` and writes:

  selection.csv         one row per child: the four metrics, the rule stages, candidate rank, selected
  selected_params.json  {"child": ..., "params": {dotted grid key: value}} with the grid's JSON types
  sweep_metrics.png     every metric against every swept hyperparameter, selected value marked
  sweep_tradeoff.png    quantization error vs distance preservation, selected child highlighted

The `check` command proves configs/som.json carries exactly the recorded selections:

    python -m tp4.experiments.som_sweep_report check configs/som.json \\
        runs/som_design_sweep runs/som_sweep --run runs/som

This module must not import tp4.experiments.som or the registry (the som module imports it).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Sequence
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
from ..config import load_config
from ..plotting import plt, save_figure
from ..runs import PROJECT_ROOT, RunContext, write_csv, write_json

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


def _neuron_label(n: float) -> str:
    root = math.isqrt(int(n)) if np.isfinite(n) else 0
    return f"{root}x{root}" if np.isfinite(n) and root * root == int(n) else f"{int(n)} neuronas"


def _plot_tradeoff(sel_table: pd.DataFrame, selected_position: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sizes = sorted(sel_table["n_neurons"].dropna().unique())
    palette = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, max(len(sizes), 2)))
    qe = sel_table["quantization_error"].to_numpy(dtype=float)
    dp = sel_table["distance_preservation"].to_numpy(dtype=float)
    a_ok = sel_table["stage_occupancy"].to_numpy(dtype=bool)
    b_ok = sel_table["stage_topology"].to_numpy(dtype=bool)
    for color, size in zip(palette, sizes):
        of_size = (sel_table["n_neurons"] == size).to_numpy()
        hollow = of_size & ~a_ok
        crossed = of_size & a_ok & ~b_ok
        kept = of_size & b_ok
        ax.scatter(qe[kept], dp[kept], s=34, color=color, label=_neuron_label(size))
        ax.scatter(qe[hollow], dp[hollow], s=34, facecolors="none", edgecolors=color, linewidths=1.1)
        ax.scatter(qe[crossed], dp[crossed], s=40, marker="x", color=color, linewidths=1.3)
    b_dp = dp[b_ok & np.isfinite(dp)]
    if len(b_dp):
        ax.axhline(b_dp.max() - DP_SLACK, color="#444444", linestyle="--", linewidth=1.1)
        ax.text(
            ax.get_xlim()[1], b_dp.max() - DP_SLACK, f" umbral de preservación ({DP_SLACK:g} bajo el máximo)",
            ha="right", va="bottom", fontsize=8, color="#444444",
        )
    row = sel_table.iloc[selected_position]
    if np.isfinite(qe[selected_position]) and np.isfinite(dp[selected_position]):
        ax.scatter([qe[selected_position]], [dp[selected_position]], s=320, marker="*", color="#D55E00", edgecolors="black", zorder=5, label="elegido")
        ax.annotate(
            str(row["child"]), (qe[selected_position], dp[selected_position]),
            textcoords="offset points", xytext=(10, 10), fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#444444"},
        )
    ax.plot([], [], "o", color="#777777", label=f"cumple ocupación (<= {MAX_EMPTY_FRAC:g} vacías) y topología")
    ax.plot([], [], "x", color="#777777", label="cumple ocupación, falla error topográfico")
    ax.plot([], [], "o", markerfacecolor="none", color="#777777", label="falla ocupación")
    ax.set_xlabel("error de cuantización (menor es mejor)")
    ax.set_ylabel("preservación de distancias (mayor es mejor)")
    ax.set_title("Compromiso entre error de cuantización y preservación de distancias")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
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
        _plot_tradeoff(out, sel.selected_position, ctx.artifact("sweep_tradeoff.png"))

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


# ---------------------------------------------------------------- check command


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_value(config: dict, dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return "<absent>"
        node = node[part]
    return node


def check_selection(
    config_path: str | Path, sweep_dirs: Sequence[str | Path], run_dir: str | Path | None = None
) -> list[str]:
    """Messages describing every disagreement between the config and the recorded selections.

    An empty list means configs/som.json equals the selections (and, with `run_dir`, that the
    final run reproduces the last sweep's selected metrics exactly).
    """
    messages: list[str] = []
    config = load_config(_resolve(config_path))
    last_summary: dict | None = None
    for sweep_dir in sweep_dirs:
        directory = _resolve(sweep_dir)
        selected_file = directory / "selected_params.json"
        if not selected_file.is_file():
            messages.append(f"MISSING {selected_file.as_posix()}")
            continue
        for key, value in _read_json(selected_file)["params"].items():
            have = _config_value(config, key)
            if have == "<absent>" or have != value or isinstance(have, bool) != isinstance(value, bool):
                messages.append(f"MISMATCH {sweep_dir}: {key} config={have} selected={value}")
        summary_file = directory / "summary.json"
        if summary_file.is_file():
            last_summary = _read_json(summary_file)
        else:
            messages.append(f"MISSING {summary_file.as_posix()}")
            last_summary = None
    if run_dir is not None:
        run_summary_file = _resolve(run_dir) / "summary.json"
        if not run_summary_file.is_file():
            messages.append(f"MISSING {run_summary_file.as_posix()}")
        elif last_summary is not None:
            run_summary = _read_json(run_summary_file)
            for name in ("quantization_error", "topographic_error", "distance_preservation"):
                got, want = run_summary.get(name), last_summary.get(f"selected_{name}")
                if got != want:
                    messages.append(f"METRIC {name} run={got} selected={want}")
    return messages


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tp4.experiments.som_sweep_report",
        description="Check that a SOM config equals the configurations selected by its sweeps.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="compare a config with the selected_params.json of its sweeps")
    check.add_argument("config", help="SOM config (e.g. configs/som.json)")
    check.add_argument("sweep_dirs", nargs="+", help="sweep run dirs, in order (the last one is the main sweep)")
    check.add_argument("--run", dest="run", default=None, help="final run dir whose metrics must equal the last sweep's selection")
    args = parser.parse_args(argv)

    messages = check_selection(args.config, args.sweep_dirs, args.run)
    if messages:
        for message in messages:
            print(message)
        return 1
    n_params = sum(len(_read_json(_resolve(d) / "selected_params.json")["params"]) for d in args.sweep_dirs)
    tail = ""
    if args.run is not None:
        tail = f", run reproduces {_read_json(_resolve(args.sweep_dirs[-1]) / 'summary.json')['selected_child']}"
    print(f"ok: {n_params} params match{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
