"""One-parameter comparisons using mean and sample spread across training seeds."""

from __future__ import annotations

import pandas as pd

from ..plotting import plt, save_figure
from ..runs import RunContext, write_csv

METRICS = ("mean_bmu_distance", "n_empty_neurons", "empty_neuron_frac", "u_mean", "u_max")
PLOT_METRICS = (
    ("mean_bmu_distance", "Distancia media al ganador"),
    ("empty_neuron_frac", "Fracción de neuronas vacías"),
    ("u_mean", "Distancia media entre vecinos (U)"),
)
PARAMETER_LABELS = {
    "params.rows": "Tamaño de grilla (k×k)",
    "params.eta0": "Tasa inicial η₀",
    "params.radius0": "Radio de vecindad",
    "params.epochs": "Épocas",
    "params.topology": "Topología",
    "params.init": "Inicialización",
}


def report(ctx: RunContext, table: pd.DataFrame) -> dict:
    keys = list(ctx.config["grid"])
    columns = ["child", *keys, "varied_parameter", "parameter_value", "n_repeats", "representative_seed"]
    columns += [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
    write_csv(ctx.artifact("comparison.csv"), table[columns])
    if ctx.plots:
        fig, axes = plt.subplots(len(PLOT_METRICS), len(keys), figsize=(4.2 * len(keys), 9.5), squeeze=False)
        for column, key in enumerate(keys):
            rows = table.loc[table["varied_parameter"] == key]
            positions = range(len(rows))
            values = rows["parameter_value"].tolist()
            labels = [f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool)
                      else str(value) for value in values]
            if key == "params.rows":
                labels = [f"{value}×{value}" for value in labels]
            for row, (metric, label) in enumerate(PLOT_METRICS):
                ax = axes[row, column]
                ax.errorbar(positions, rows[f"{metric}_mean"], yerr=rows[f"{metric}_std"],
                            fmt="o-", capsize=3, linewidth=1.2)
                ax.set_xticks(list(positions), labels=labels)
                ax.set_xlabel(PARAMETER_LABELS.get(key, key))
                if column == 0:
                    ax.set_ylabel(label)
                ax.grid(alpha=0.3)
        repeats = ", ".join(str(int(n)) for n in sorted(table["n_repeats"].unique()))
        fig.suptitle(f"Un parámetro por vez; los demás fijos\nMedia ±1 DE entre semillas ({repeats} repeticiones por valor)")
        fig.tight_layout()
        save_figure(fig, ctx.artifact("sweep_metrics.png"))
    return {"comparison": "one parameter at a time; mean and sample standard deviation across training seeds"}
