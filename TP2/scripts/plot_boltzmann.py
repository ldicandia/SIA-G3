"""Plot fitness vs generations for Boltzmann selection run(s).

Supports:
1. Multi-seed matrix run directory (e.g. `runs/matrix_boltzmann` or
   `runs/matrix_boltzmann/selection-boltzmann` containing `seed*` subdirectories):
   plots median best fitness with an IQR (interquartile range) confidence band
   across seeds.
2. Single run directory (containing `metrics.csv` directly):
   plots best and mean fitness across generations.

Usage:
    .venv/bin/python scripts/plot_boltzmann.py --run-dir runs/matrix_boltzmann
"""

from __future__ import annotations

import argparse
import csv
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


def _load_single_metrics(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols: dict[str, list[float]] = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "worst_fitness": [],
        }
        for row in reader:
            for k in cols:
                if k in row and row[k] != "":
                    cols[k].append(float(row[k]))
    return {k: np.array(v, dtype=np.float64) for k, v in cols.items()}


def plot_boltzmann(run_dir: Path, out_path: Path, title: str | None = None) -> Path:
    target_dir = run_dir.resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory {target_dir} does not exist.")

    # Check if this is a matrix root containing a selection-boltzmann child
    if (target_dir / "selection-boltzmann").is_dir():
        target_dir = target_dir / "selection-boltzmann"

    seed_dirs = sorted(target_dir.glob("seed*"))

    fig, ax = plt.subplots(figsize=(8, 5))

    if seed_dirs:
        # Multi-seed mode (matrix cell)
        n_seeds = len(seed_dirs)
        curves = load_seed_curves(target_dir, x_col="generation", y_col="best_fitness")
        grid_x, values = align_on_grid(curves)
        median, q1, q3 = median_iqr(values)

        # Plot individual seeds as faint lines
        for i, (x_s, y_s) in enumerate(curves):
            ax.plot(x_s, y_s, color="tab:blue", alpha=0.25, linewidth=1.0, label="Semilla individual" if i == 0 else None)

        # Plot median and IQR band
        ax.plot(grid_x, median, color="tab:blue", linewidth=2.0, label=f"Mediana (n={n_seeds})")
        ax.fill_between(grid_x, q1, q3, color="tab:blue", alpha=0.25, label="Rango Intercuartil (IQR)")

        default_title = f"Selección Boltzmann (Padres: Boltzmann, Reemplazo: Élite, n={n_seeds})\nFitness en función de Generaciones"
    elif (target_dir / "metrics.csv").is_file():
        # Single run mode
        data = _load_single_metrics(target_dir / "metrics.csv")
        gens = data["generation"]
        ax.plot(gens, data["best_fitness"], color="tab:blue", linewidth=2.0, label="Mejor Fitness")
        if len(data["mean_fitness"]) > 0:
            ax.plot(gens, data["mean_fitness"], color="tab:orange", linewidth=1.5, linestyle="--", label="Fitness Medio")

        default_title = "Selección Boltzmann (Padres: Boltzmann, Reemplazo: Élite)\nFitness en función de Generaciones"
    else:
        raise ValueError(
            f"Could not find seeds (seed*/metrics.csv) or a single metrics.csv under {target_dir}"
        )

    ax.set_xlabel("Generación", fontsize=11)
    ax.set_ylabel("Fitness", fontsize=11)
    ax.set_title(title or default_title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot Boltzmann selection fitness vs generations.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/matrix_boltzmann"),
        help="Path to matrix run directory (runs/matrix_boltzmann) or single run directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots/fig_boltzmann_fitness_generations.png"),
        help="Output image path (default: plots/fig_boltzmann_fitness_generations.png)",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional custom plot title")

    args = parser.parse_args(argv)
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    out_path = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out

    out = plot_boltzmann(run_dir, out_path, title=args.title)
    print(f"Plot saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
