"""Plot all 7 selection methods together, demonstrating that with fitness scaling
and temperature calibration, ALL methods converge (>0.96) even when applied
symmetrically to BOTH parent selection and generational replacement (no elitism).

Generates two figures:
1. `plots/fig_all_methods_scaled.png`:
   Single unified figure with all 7 methods converging to >0.96, with unscaled
   proportional baselines shown as dashed reference lines near the bottom.
2. `plots/fig_selection_before_after.png`:
   Side-by-side 2-panel comparison:
   - Left: Canonical baseline (Ruleta/SUS/Boltzmann flat at ~0.72).
   - Right: Scaled & calibrated (All 7 methods converging to 0.96 - 0.98).

Usage:
    .venv/bin/python scripts/plot_all_methods_scaled.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tp2.experiments.aggregate import align_on_grid, load_seed_curves, median_iqr

# Configuration of all 7 methods
SCALED_METHODS = [
    ("Ranking", "runs/matrix/selection-ranking", "#2ca02c", "-", 2.0),
    ("Torneo Det.", "runs/matrix/selection-tournament_deterministic", "#1f77b4", "-", 1.8),
    ("Torneo Prob.", "runs/matrix/selection-tournament_probabilistic", "#17becf", "-", 1.8),
    ("Elite", "runs/matrix/selection-elite", "#9467bd", "-", 1.8),
    ("Universal (Sigma)", "runs/matrix_scaling/scaling-universal-sigma", "#ff7f0e", "-", 2.2),
    ("Boltzmann (Tc=0.002)", "runs/matrix_bsweep/bsweep-t0_0.15-tc_0.002-k_0.01", "#e377c2", "-", 2.2),
    ("Ruleta (Sigma)", "runs/matrix_scaling/scaling-roulette-sigma", "#d62728", "-", 2.2),
]

UNSCALED_PROPORTIONAL = [
    ("Ruleta cruda (sin escalar)", "runs/matrix/selection-roulette", "#d62728", ":", 1.5),
    ("Universal cruda (sin escalar)", "runs/matrix/selection-universal", "#ff7f0e", ":", 1.5),
    ("Boltzmann crudo (Tc=0.5)", "runs/matrix/selection-boltzmann", "#e377c2", ":", 1.5),
]


def plot_unified(out_path: Path) -> None:
    """Single plot with all 7 methods converging + unscaled references below."""
    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.axhline(0.695, color="black", linestyle="--", linewidth=1.0, alpha=0.5, label="Piso canvas vacío (0.695)")

    # Plot unscaled reference curves (faint/dashed)
    for label, path_str, color, ls, lw in UNSCALED_PROPORTIONAL:
        p = PROJECT_ROOT / path_str
        # Fail loudly: a figure that claims to compare 7 methods and quietly
        # renders 5 is worse than no figure at all.
        if not p.is_dir():
            raise FileNotFoundError(f"missing cell {p} for {label!r} -- run the matrix that produces it first")
        curves = load_seed_curves(p, x_col="generation", y_col="best_fitness")
        grid_x, values = align_on_grid(curves)
        med, _, _ = median_iqr(values)
        ax.plot(grid_x, med, color=color, linestyle=ls, linewidth=lw, alpha=0.6, label=f"{label} -> {med[-1]:.4f}")

    # Plot all 7 scaled/working methods (solid lines + IQR bands)
    for label, path_str, color, ls, lw in SCALED_METHODS:
        p = PROJECT_ROOT / path_str
        # Fail loudly: a figure that claims to compare 7 methods and quietly
        # renders 5 is worse than no figure at all.
        if not p.is_dir():
            raise FileNotFoundError(f"missing cell {p} for {label!r} -- run the matrix that produces it first")
        curves = load_seed_curves(p, x_col="generation", y_col="best_fitness")
        grid_x, values = align_on_grid(curves)
        med, q1, q3 = median_iqr(values)
        ax.plot(grid_x, med, color=color, linestyle=ls, linewidth=lw, label=f"{label} -> {med[-1]:.4f}")
        ax.fill_between(grid_x, q1, q3, color=color, alpha=0.12)

    ax.set_xlabel("Generaciones", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mejor Fitness", fontsize=12, fontweight="bold")
    ax.set_title(
        "Convergencia de los 7 Métodos de Selección con Reescalado de Fitness y Tc Calibrado\n"
        "(Padres y Reemplazo usan el mismo método sin elitismo · n=5 semillas por celda)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(0.65, 1.0)
    ax.set_xlim(0, 3000)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9.5, borderaxespad=0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_before_after(out_path: Path) -> None:
    """Side-by-side comparison: Canonical vs Scaled/Calibrated."""
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Panel 1: Canonical baseline (from runs/matrix)
    canonical = [
        ("Ranking", "runs/matrix/selection-ranking", "#2ca02c"),
        ("Elite", "runs/matrix/selection-elite", "#9467bd"),
        ("Torneo Det.", "runs/matrix/selection-tournament_deterministic", "#1f77b4"),
        ("Torneo Prob.", "runs/matrix/selection-tournament_probabilistic", "#17becf"),
        ("Universal (SUS)", "runs/matrix/selection-universal", "#ff7f0e"),
        ("Boltzmann", "runs/matrix/selection-boltzmann", "#e377c2"),
        ("Ruleta", "runs/matrix/selection-roulette", "#d62728"),
    ]

    ax_before.axhline(0.695, color="black", linestyle="--", linewidth=1.0, alpha=0.5, label="Canvas vacío (~0.695)")
    for label, path_str, color in canonical:
        p = PROJECT_ROOT / path_str
        curves = load_seed_curves(p, x_col="generation", y_col="best_fitness")
        grid_x, values = align_on_grid(curves)
        med, q1, q3 = median_iqr(values)
        ax_before.plot(grid_x, med, color=color, linewidth=2.0, label=f"{label} ({med[-1]:.4f})")
        ax_before.fill_between(grid_x, q1, q3, color=color, alpha=0.12)

    ax_before.set_xlabel("Generaciones", fontsize=11, fontweight="bold")
    ax_before.set_ylabel("Mejor Fitness", fontsize=11, fontweight="bold")
    ax_before.set_title("Línea Base Canónica (Sin Reescalar)\nRuleta, SUS y Boltzmann se estancan por el offset", fontsize=12, fontweight="bold")
    ax_before.grid(True, linestyle=":", alpha=0.6)
    ax_before.legend(loc="lower right", fontsize=8.5)
    ax_before.set_ylim(0.65, 1.0)
    ax_before.set_xlim(0, 3000)

    # Panel 2: Scaled & calibrated
    ax_after.axhline(0.695, color="black", linestyle="--", linewidth=1.0, alpha=0.5, label="Canvas vacío (~0.695)")
    for label, path_str, color, _, lw in SCALED_METHODS:
        p = PROJECT_ROOT / path_str
        curves = load_seed_curves(p, x_col="generation", y_col="best_fitness")
        grid_x, values = align_on_grid(curves)
        med, q1, q3 = median_iqr(values)
        ax_after.plot(grid_x, med, color=color, linewidth=lw, label=f"{label} ({med[-1]:.4f})")
        ax_after.fill_between(grid_x, q1, q3, color=color, alpha=0.12)

    ax_after.set_xlabel("Generaciones", fontsize=11, fontweight="bold")
    ax_after.set_title("Con Reescalado y Tc Calibrado (Sin Elitismo)\nTodos los 7 métodos convergen a > 0.96", fontsize=12, fontweight="bold")
    ax_after.grid(True, linestyle=":", alpha=0.6)
    ax_after.legend(loc="lower right", fontsize=8.5)
    ax_after.set_xlim(0, 3000)

    fig.suptitle(
        "Impacto del Reescalado de Fitness en los Métodos de Selección (n=5)\n"
        "Mismo método en Padres y Reemplazo: sin élite, la ruleta y SUS alcanzan el mismo nivel que Ranking y Torneo",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    out_unified = PROJECT_ROOT / "plots" / "fig_all_methods_scaled.png"
    out_before_after = PROJECT_ROOT / "plots" / "fig_selection_before_after.png"

    plot_unified(out_unified)
    print(f"Unified plot saved to: {out_unified}")

    plot_before_after(out_before_after)
    print(f"Before/After comparison saved to: {out_before_after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
