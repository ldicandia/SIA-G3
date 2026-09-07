"""Plot fitness vs generations comparing fitness scaling modes for Roulette and SUS.

Reads runs from `runs/matrix_scaling/` and plots median curves with IQR bands:
- Left panel: Roulette (none vs offset vs sigma)
- Right panel: Universal / SUS (none vs offset vs sigma)

Usage:
    .venv/bin/python scripts/plot_scaling.py
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


MODES = [
    ("none", "Sin escalar (crudo)", "tab:gray", ":"),
    ("offset", "Offset (f - f_min)", "tab:blue", "--"),
    ("sigma", "Sigma scaling (f - (μ - 2σ))", "tab:red", "-"),
]

_RUN_HINT = (
    "produce the runs first: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    ".venv/bin/python -m tp2.experiments.runner --spec configs/experiments/scaling_matrix.json "
    "--out runs/matrix_scaling --jobs 8"
)


def plot_scaling(
    matrix_dir: Path,
    out_path: Path,
    title: str | None = None,
) -> Path:
    matrix_dir = matrix_dir.resolve()
    if not matrix_dir.exists():
        raise FileNotFoundError(f"Matrix dir {matrix_dir} does not exist.")

    fig, (ax_roulette, ax_sus) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    selectors = [
        ("roulette", "Ruleta (Padres: Ruleta, Reemplazo: Ruleta)", ax_roulette),
        ("universal", "Universal / SUS (Padres: SUS, Reemplazo: SUS)", ax_sus),
    ]

    for sel_key, sel_name, ax in selectors:
        # Reference baseline: blank canvas
        ax.axhline(0.695, color="black", linestyle=":", linewidth=1.2, alpha=0.6, label="Piso canvas vacío (0.695)")

        for mode_key, mode_label, color, linestyle in MODES:
            cell_dir = matrix_dir / f"scaling-{sel_key}-{mode_key}"
            # Fail loudly on a missing cell rather than quietly dropping a
            # curve: a figure that silently renders with fewer modes than it
            # claims to compare is exactly the failure this project's atomic
            # metrics.csv write and _require_dir hint exist to prevent.
            if not cell_dir.is_dir():
                raise FileNotFoundError(f"missing cell {cell_dir} -- {_RUN_HINT}")

            curves = load_seed_curves(cell_dir, x_col="generation", y_col="best_fitness")
            if not curves:
                raise FileNotFoundError(f"no completed seeds under {cell_dir} -- {_RUN_HINT}")

            grid_x, values = align_on_grid(curves)
            median, q1, q3 = median_iqr(values)
            n_seeds = len(curves)

            final_med = median[-1]
            label = f"{mode_label}  ->  {final_med:.4f}"

            ax.plot(
                grid_x,
                median,
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=label,
            )
            ax.fill_between(grid_x, q1, q3, color=color, alpha=0.18)

        ax.set_xlabel("Generaciones", fontsize=11, fontweight="bold")
        ax.set_title(sel_name, fontsize=12, fontweight="bold", pad=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
        ax.set_ylim(0.65, 1.0)
        ax.set_xlim(0, 3000)

    ax_roulette.set_ylabel("Mejor Fitness", fontsize=11, fontweight="bold")

    fig.suptitle(
        title
        or "Reescalado de Fitness en Operadores Proporcionales (Sin Elitismo)\n"
        "Padres y Reemplazo usan el mismo método. Al eliminar el offset base (~0.695), "
        "Ruleta y SUS pasan de ~0.72 a 0.95-0.97 (n=5)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot fitness scaling comparative curves.")
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=Path("runs/matrix_scaling"),
        help="Path to matrix scaling runs directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots/fig_scaling_comparison.png"),
        help="Output image path",
    )
    args = parser.parse_args(argv)

    matrix_dir = args.matrix_dir if args.matrix_dir.is_absolute() else PROJECT_ROOT / args.matrix_dir
    out_path = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out

    out = plot_scaling(matrix_dir, out_path)
    print(f"Plot saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
