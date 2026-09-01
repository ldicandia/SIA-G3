"""Build the five named, honestly-captioned comparative figures the
presentation cites, reading pre-existing matrix and hill-climber run output.

Never re-runs the GA -- every figure is read back from `runs/matrix/`
(`python -m tp2.experiments.runner`'s output) and `runs/hillclimber/`
(`python -m tp2.baselines.hillclimber`'s output).

`matplotlib.use("Agg")` MUST be set before any `pyplot` import: this WSL box
has no display, and the wrong backend crashes on import rather than on save.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Runnable directly (`python scripts/generate_plots.py`), not only via
# `python -m`: put the project root on sys.path before importing tp2, the
# same way tp2/cli.py's own PROJECT_ROOT-relative-path convention resolves
# paths, so this script needs no PYTHONPATH set by the caller.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import argparse  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from tp2.experiments.aggregate import align_on_grid, load_seed_curves, median_iqr  # noqa: E402

# The figure-to-claim mapping made literal, in-repo data -- not only prose in
# a plan. Every figure's title includes this string verbatim plus its own
# stated `n=` replicate count (Pitfall 16's "state the replicate count on
# every figure").
FIGURE_CLAIMS: dict[str, str] = {
    "fig_selection_fitness.png": (
        "Evidence for: selection-pressure differences are real and visible "
        "at equal render budget across all 7 registered methods"
    ),
    "fig_selection_diversity.png": (
        "Evidence for EXP-04: diversity collapse under high-pressure "
        "selectors vs its preservation under low-pressure ones, supporting "
        "the premature-convergence analysis"
    ),
    "fig_survival_kn.png": (
        "Evidence for Phase 3 Success Criterion 2, now aggregated across "
        "seeds: additive survival's best-fitness curve stays monotone while "
        "exclusive survival's genuinely dips under K>N"
    ),
    "fig_crossover_control.png": (
        "Evidence for/against Pitfall 6's crossover-destructiveness "
        "prediction: baseline crossover vs a mutation-only control at "
        "equal render budget"
    ),
    "fig_hillclimber_comparison.png": (
        "Evidence for ROADMAP Success Criterion 5: the (1+1) hill climber "
        "vs the best-performing GA configuration at equal render budget, "
        "reported honestly whichever way it goes"
    ),
}


class GeneratePlotsError(ValueError):
    """A referenced run directory is missing or malformed."""


def plot_arm(
    cell_dirs: dict[str, Path],
    title: str,
    out_path: Path,
    y_col: str = "best_fitness",
    y_label: str = "Best fitness",
    n_seeds: int = 5,
) -> None:
    """One figure: median + IQR band per labeled cell, all on one axes.

    `cell_dirs` is iterated in a STABLE, explicit order (sorted keys) so the
    legend order is reproducible across runs, never dependent on dict-
    construction order.
    """
    fig, ax = plt.subplots()
    for label in sorted(cell_dirs):
        curves = load_seed_curves(cell_dirs[label], y_col=y_col)
        grid_x, values = align_on_grid(curves)
        median, q1, q3 = median_iqr(values)
        ax.plot(grid_x, median, label=label)
        ax.fill_between(grid_x, q1, q3, alpha=0.2)
    ax.set_xlabel("Renders (cumulative fitness evaluations)")
    ax.set_ylabel(y_label)
    # The "n=" here is this plan's own omitted-n prohibition's concrete
    # enforcement, not just a described intention.
    ax.set_title(f"{title}  (n={n_seeds})")
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build comparative figures from pre-existing matrix run output. "
            "Never re-runs the GA."
        )
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=Path("runs/_matrix_tracer"),
        help="root directory of matrix cell run directories (Task 1 tracer scope)",
    )
    parser.add_argument(
        "--plots-dir", type=Path, default=Path("plots"), help="output directory for generated figures"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Task 1's tracer scope: the "selection" arm's cells, one figure.

    Task 2 extends this to the full five-figure set from real matrix and
    hill-climber output; this task's job is proving the ONE-figure path
    works end to end against real data with the correct x-axis semantics.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    matrix_root = args.matrix_root if args.matrix_root.is_absolute() else PROJECT_ROOT / args.matrix_root
    plots_dir = args.plots_dir if args.plots_dir.is_absolute() else PROJECT_ROOT / args.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    cell_dirs = {cell_dir.name.removeprefix("selection-"): cell_dir for cell_dir in sorted(matrix_root.glob("selection-*"))}
    if not cell_dirs:
        raise GeneratePlotsError(f"no selection-* cells found under {matrix_root}")

    n_seeds = len(load_seed_curves(next(iter(cell_dirs.values()))))
    out_path = plots_dir / "fig_selection_fitness.png"
    plot_arm(cell_dirs, FIGURE_CLAIMS["fig_selection_fitness.png"], out_path, n_seeds=n_seeds)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
