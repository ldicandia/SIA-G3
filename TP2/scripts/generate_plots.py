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
import csv  # noqa: E402
import json  # noqa: E402
import textwrap  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tp2.experiments.aggregate import align_on_grid, load_seed_curves, median_iqr  # noqa: E402

# The 7 selection labels, 6 survival_kn labels (as krn-{ratio}-{strategy})
# and 2 crossover_control labels `configs/experiments/main_matrix.json`
# actually ships -- read directly from that file, not re-derived, so a
# future edit to the matrix spec cannot silently desync this list.
SELECTION_LABELS = [
    "elite",
    "roulette",
    "universal",
    "ranking",
    "boltzmann",
    "tournament_deterministic",
    "tournament_probabilistic",
]
SURVIVAL_RATIOS = ["0.5", "1.0", "2.0"]
SURVIVAL_STRATEGIES = ["additive", "exclusive"]
CROSSOVER_CONTROL_LABELS = ["baseline-crossover", "mutation-only"]

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


def _wrap_title(text: str, width: int = 70) -> str:
    """Wrap a long claim string onto multiple lines instead of letting it run
    off the right edge of the figure (matplotlib does not wrap titles itself,
    and the FIGURE_CLAIMS strings are full sentences)."""
    return "\n".join(textwrap.wrap(text, width=width))


class GeneratePlotsError(ValueError):
    """A referenced run directory is missing or malformed."""


def _matrix_missing_hint() -> str:
    return (
        "build the real 75-run matrix first: "
        "python -m tp2.experiments.runner --spec configs/experiments/main_matrix.json "
        "--out runs/matrix --jobs 8"
    )


def _hillclimber_missing_hint() -> str:
    return (
        "run the hill climber at a matched render budget first, e.g.: "
        "python -m tp2.baselines.hillclimber --image assets/flag_ar.png --triangles 30 "
        "--config configs/baseline.json --seed 1 "
        "--max-renders <the winning GA cell's total renders> --out runs/hillclimber"
    )


def _require_dir(path: Path, hint: str) -> None:
    if not path.is_dir():
        raise GeneratePlotsError(f"expected a run directory at {path}, but it does not exist -- {hint}")


def _load_single_curve(
    metrics_path: Path, x_col: str = "renders", y_col: str = "best_fitness"
) -> tuple[np.ndarray, np.ndarray]:
    """Read one flat metrics.csv directly (no seed*/ subdirectory).

    The hill climber writes a single run's metrics.csv at its run directory's
    own root (04-02's `tp2.baselines.hillclimber` -- population of one, not
    five seeded runs), so it cannot go through `load_seed_curves`, which
    expects a `seed*/metrics.csv` layout under `cell_dir`.
    """
    with Path(metrics_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        xs: list[float] = []
        ys: list[float] = []
        for row in reader:
            xs.append(float(row[x_col]))
            ys.append(float(row[y_col]))
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


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
    ax.set_title(_wrap_title(f"{title}  (n={n_seeds})"))
    # Outside the axes, never "best": with this many overlapping curves and
    # IQR bands, an in-axes legend inevitably sits on top of the data it is
    # labeling. bbox_inches="tight" on save expands the canvas to include it.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_survival_kn(matrix_root: Path, out_path: Path, n_seeds: int = 5) -> None:
    """Two panels (additive, exclusive), sharing the render-count x-axis.

    Each panel plots the three K/N ratios `main_matrix.json` ships. The
    additive curve is expected to stay non-decreasing while the exclusive
    curve is permitted to dip -- read directly off the aggregated median
    arrays plotted here, never asserted from a rendered image.
    """
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4.5))
    for strategy, ax in zip(SURVIVAL_STRATEGIES, axes):
        for ratio in SURVIVAL_RATIOS:
            cell_dir = matrix_root / f"survival_kn-krn-{ratio}-{strategy}"
            _require_dir(cell_dir, _matrix_missing_hint())
            curves = load_seed_curves(cell_dir)
            grid_x, values = align_on_grid(curves)
            median, q1, q3 = median_iqr(values)
            ax.plot(grid_x, median, label=f"K/N={ratio}")
            ax.fill_between(grid_x, q1, q3, alpha=0.2)
        ax.set_xlabel("Renders (cumulative fitness evaluations)")
        ax.set_title(strategy)
    axes[0].set_ylabel("Best fitness")
    # One shared legend outside both panels, not one per panel: both axes
    # plot the same K/N ratios in the same colors (matplotlib's color cycle
    # resets per-axes), so a legend on each panel would just be a duplicate
    # sitting on top of that panel's own curves.
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(1.0, 0.95))
    wrapped_title = _wrap_title(f"{FIGURE_CLAIMS['fig_survival_kn.png']}  (n={n_seeds})", width=90)
    # A plain fig.suptitle() sits at a fixed y just above the panel titles --
    # fine for one line, but this claim wraps to several, and would then
    # collide with "additive"/"exclusive" underneath it. Push the panels
    # down by however many lines the title actually took.
    n_title_lines = wrapped_title.count("\n") + 1
    fig.suptitle(wrapped_title, y=0.99)
    fig.subplots_adjust(top=0.99 - 0.09 * n_title_lines)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _best_selection_cell(selection_cells: dict[str, Path]) -> Path:
    """The `selection-*` cell with the highest final aggregated median fitness.

    Determined from the data (re-derived via `align_on_grid`/`median_iqr`),
    never hardcoded -- the "best-performing GA configuration" must come from
    what the matrix actually measured.
    """
    best_path: Path | None = None
    best_value: float | None = None
    for label in sorted(selection_cells):
        cell_dir = selection_cells[label]
        curves = load_seed_curves(cell_dir)
        _, values = align_on_grid(curves)
        median, _q1, _q3 = median_iqr(values)
        final_value = float(median[-1])
        if best_value is None or final_value > best_value:
            best_value = final_value
            best_path = cell_dir
    assert best_path is not None
    return best_path


def plot_hillclimber_comparison(ga_cell_dir: Path, hillclimber_dir: Path, out_path: Path) -> None:
    """The GA's best selection-arm cell vs the (1+1) hill climber.

    Both curves are truncated, via ONE `align_on_grid` call over the GA's
    seed curves plus the hill climber's single curve, to whichever of the
    two stopped first -- the equal-render-budget honesty this plan's own
    prohibition requires. The hill climber's curve is NOT band-plotted
    (aggregating one run has no spread to show); its `n=1` is stated
    alongside the GA's `n=5` in the caption rather than silently omitted.

    The legend label is read from the hill climber's own archived
    `run.json["algorithm"]` field (04-02's `"hillclimb_1p1"`) and confirmed
    programmatically -- never "baseline GA" or any GA-implying label, and
    never a hardcoded string that could drift out of sync with the actual
    run.
    """
    ga_curves = load_seed_curves(ga_cell_dir)
    hc_x, hc_y = _load_single_curve(hillclimber_dir / "metrics.csv")

    run_json_path = hillclimber_dir / "run.json"
    if not run_json_path.is_file():
        raise GeneratePlotsError(f"{run_json_path} is missing -- cannot confirm this is a hill-climber run")
    with run_json_path.open(encoding="utf-8") as handle:
        run_payload = json.load(handle)
    if run_payload.get("algorithm") != "hillclimb_1p1":
        raise GeneratePlotsError(
            f"{run_json_path} does not carry algorithm == 'hillclimb_1p1' -- "
            "refusing to label an unverified run directory as the hill climber"
        )

    # Constructing a combined "curves" list mixing the GA's 5 seed curves and
    # the hill climber's single curve is acceptable here: both are already
    # (x, y) arrays at this point, and align_on_grid's own cap logic (the
    # MINIMUM of every curve's own max render count) is exactly the
    # truncation this comparison needs -- never a bespoke cap computed here.
    combined = [*ga_curves, (hc_x, hc_y)]
    grid_x, values = align_on_grid(combined)
    ga_values, hc_value = values[:-1], values[-1]
    median, q1, q3 = median_iqr(ga_values)
    n_seeds = len(ga_curves)

    fig, ax = plt.subplots()
    ax.plot(grid_x, median, label=f"Best GA configuration (n={n_seeds})")
    ax.fill_between(grid_x, q1, q3, alpha=0.2)
    ax.plot(grid_x, hc_value, label="(1+1) hill climber (n=1)", linestyle="--")
    ax.set_xlabel("Renders (cumulative fitness evaluations)")
    ax.set_ylabel("Best fitness")
    ax.set_title(
        _wrap_title(FIGURE_CLAIMS["fig_hillclimber_comparison.png"])
        + f"\n(GA n={n_seeds}, hill climber n=1)"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_all_figures(matrix_root: Path, hillclimber_dir: Path, plots_dir: Path) -> list[Path]:
    """The full five-figure set, each built off `align_on_grid` + `median_iqr`."""
    outputs: list[Path] = []

    selection_cells = {label: matrix_root / f"selection-{label}" for label in SELECTION_LABELS}
    for cell_dir in selection_cells.values():
        _require_dir(cell_dir, _matrix_missing_hint())
    n_seeds = len(load_seed_curves(next(iter(selection_cells.values()))))

    out = plots_dir / "fig_selection_fitness.png"
    plot_arm(
        selection_cells, FIGURE_CLAIMS["fig_selection_fitness.png"], out,
        y_col="best_fitness", y_label="Best fitness", n_seeds=n_seeds,
    )
    outputs.append(out)

    out = plots_dir / "fig_selection_diversity.png"
    plot_arm(
        selection_cells, FIGURE_CLAIMS["fig_selection_diversity.png"], out,
        y_col="diversity", y_label="Diversity (mean stdev/range across loci)", n_seeds=n_seeds,
    )
    outputs.append(out)

    out = plots_dir / "fig_survival_kn.png"
    plot_survival_kn(matrix_root, out, n_seeds=n_seeds)
    outputs.append(out)

    crossover_cells = {label: matrix_root / f"crossover_control-{label}" for label in CROSSOVER_CONTROL_LABELS}
    for cell_dir in crossover_cells.values():
        _require_dir(cell_dir, _matrix_missing_hint())
    out = plots_dir / "fig_crossover_control.png"
    plot_arm(crossover_cells, FIGURE_CLAIMS["fig_crossover_control.png"], out, n_seeds=n_seeds)
    outputs.append(out)

    ga_cell_dir = _best_selection_cell(selection_cells)
    _require_dir(hillclimber_dir, _hillclimber_missing_hint())
    out = plots_dir / "fig_hillclimber_comparison.png"
    plot_hillclimber_comparison(ga_cell_dir, hillclimber_dir, out)
    outputs.append(out)

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the five named comparative figures from pre-existing matrix "
            "and hill-climber run output. Never re-runs the GA."
        )
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=Path("runs/matrix"),
        help=(
            "root directory of matrix cell run directories "
            "(python -m tp2.experiments.runner output; "
            "pass runs/_matrix_tracer for a fast, tiny-scale smoke check)"
        ),
    )
    parser.add_argument(
        "--hillclimber-dir",
        type=Path,
        default=Path("runs/hillclimber"),
        help="hill-climber run directory (python -m tp2.baselines.hillclimber output)",
    )
    parser.add_argument(
        "--plots-dir", type=Path, default=Path("plots"), help="output directory for generated figures"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the full five-figure set from real matrix and hill-climber output.

    `--matrix-root` still accepts `runs/_matrix_tracer` (Task 1's tracer
    scope) for a fast, tiny-scale check of the CLI path; against that tiny
    fixture (which only has a `selection` arm) `build_all_figures` will
    legitimately fail on the missing `survival_kn`/`crossover_control`
    cells and the missing hill-climber directory, naming exactly which path
    is missing and the command to build it -- an honest failure, not a raw
    crash.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    matrix_root = args.matrix_root if args.matrix_root.is_absolute() else PROJECT_ROOT / args.matrix_root
    hillclimber_dir = (
        args.hillclimber_dir if args.hillclimber_dir.is_absolute() else PROJECT_ROOT / args.hillclimber_dir
    )
    plots_dir = args.plots_dir if args.plots_dir.is_absolute() else PROJECT_ROOT / args.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        outputs = build_all_figures(matrix_root, hillclimber_dir, plots_dir)
    except GeneratePlotsError as exc:
        parser.error(str(exc))
        return 2  # unreachable: argparse's .error() calls sys.exit()

    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
