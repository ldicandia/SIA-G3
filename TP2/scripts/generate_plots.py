"""Build the seven named, honestly-captioned comparative figures the
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
from tp2.engine.fitness import Evaluator  # noqa: E402
from tp2.engine.genome import chromosome_length  # noqa: E402
from tp2.io.images import load_target  # noqa: E402

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

# The 8 mutation labels `configs/experiments/mutation_matrix.json` ships, in
# the order the figure should read them: the four cátedra scopes at the
# baseline's own Pm=0.9 first, then the three rate-matched controls, then the
# Michalewicz schedule. Same-Pm and rate-matched cells are NOT interchangeable
# -- see MUTATION_EXPECTED_GENES below for why.
MUTATION_LABELS = [
    "gene",
    "multigen_limited",
    "multigen_uniform",
    "complete",
    "matched-multigen_limited",
    "matched-multigen_uniform",
    "matched-complete",
    "gene-non_uniform",
]

# Expected number of genes mutated per child for each cell, on this matrix's
# 30-triangle chromosome (L = 11*30 = 330 loci). Derived from each operator's
# own definition in tp2/engine/operators/mutation.py, NOT measured:
#   gene              -> Pm                      (one locus, with prob Pm)
#   multigen_limited  -> ((m+1)/2) * Pm          (count drawn in [1,m])
#   multigen_uniform  -> L * Pm                  (each locus independently)
#   complete          -> L * Pm                  (all L loci, with prob Pm)
# This is the whole point of the arm: at the SAME Pm=0.9 the four scopes
# mutate 0.9, 2.7, 297 and 297 genes per child -- a 330x spread. Comparing
# them at equal Pm compares four different search radii, not four operators,
# which is why the matched-* cells exist.
MUTATION_EXPECTED_GENES = {
    "gene": 0.9,
    "multigen_limited": 2.7,
    "multigen_uniform": 297.0,
    "complete": 297.0,
    "matched-multigen_limited": 0.9,
    "matched-multigen_uniform": 0.9,
    "matched-complete": 0.9,
    "gene-non_uniform": 0.9,
}

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
    "fig_selection_parents_fitness.png": (
        "Evidence for: the same 7 selection methods applied to parent "
        "selection ONLY, against a common elite replacement -- the controlled "
        "counterpart to fig_selection_fitness.png, where the method also "
        "governs replacement and so nothing retains the best individual, "
        "isolating the selection operator's own effect"
    ),
    "fig_selection_parents_diversity.png": (
        "Evidence for EXP-04 with the selection operator isolated: the "
        "diversity trace of the same 7 methods applied to parent selection "
        "ONLY, against a common elite replacement, separating each method's "
        "own effect on diversity from the replacement scheme's"
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
    "fig_mutation_fitness.png": (
        "Evidence for: the four catedra mutation scopes are only comparable "
        "once their per-child mutation RATE is matched -- at the baseline's "
        "own Pm=0.9 the same nominal probability means 0.9 genes per child "
        "for gene and 297 for multigen_uniform/complete"
    ),
    "fig_mutation_final_bars.png": (
        "Evidence for: final best fitness per mutation cell, with the "
        "expected genes-mutated-per-child annotated on each bar so the "
        "same-Pm cells cannot be misread as a like-for-like comparison"
    ),
    "fig_selection_final_bars.png": (
        "Evidence for: final best fitness per selection method as a direct "
        "ranking, the endpoint of fig_selection_fitness.png's curves"
    ),
    "fig_cost.png": (
        "Evidence for: what each selection method COSTS -- total wall-clock "
        "seconds at a fixed render budget, and generations needed to reach "
        "fitness 0.95, with methods that never reach it marked as such "
        "rather than silently dropped"
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
        "build the real 110-run matrix first: "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "python -m tp2.experiments.runner --spec configs/experiments/main_matrix.json "
        "--out runs/matrix --jobs 8 "
        "(the thread pins are REQUIRED, not tuning: the runner forks a process pool and "
        "OpenMP is not fork-safe, so unpinned BLAS threads make the matrix roughly 85x "
        "slower -- effectively unrunnable -- and raising --jobs without pinning makes it "
        "worse still)"
    )


def _mutation_matrix_missing_hint() -> str:
    return (
        "build the mutation arm first: "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "python -m tp2.experiments.runner --spec configs/experiments/mutation_matrix.json "
        "--out runs/matrix_mutation --jobs 8 "
        "(same thread pins as the main matrix, and for the same fork-safety reason)"
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
    x_col: str = "generation",
    y_col: str = "best_fitness",
    x_label: str = "Generaciones (igual presupuesto de renders)",
    y_label: str = "Best fitness",
    n_seeds: int = 5,
) -> None:
    """One figure: median + IQR band per labeled cell, all on one axes.

    `cell_dirs` is iterated in a STABLE, explicit order (sorted keys) so the
    legend order is reproducible across runs, never dependent on dict-
    construction order.

    Defaults to a generation x-axis, which stays an honest equal-render-budget
    comparison ONLY because every cell passed here shares one `children`
    count, making renders a fixed multiple of generations. An arm that varies
    `children_ratio` must pass `x_col="renders"` instead -- see
    `plot_survival_kn`, which does.
    """
    fig, ax = plt.subplots()
    for label in sorted(cell_dirs):
        curves = load_seed_curves(cell_dirs[label], x_col=x_col, y_col=y_col)
        grid_x, values = align_on_grid(curves)
        median, q1, q3 = median_iqr(values)
        ax.plot(grid_x, median, label=label)
        ax.fill_between(grid_x, q1, q3, alpha=0.2)
    ax.set_xlabel(x_label)
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


def plot_survival_kn(
    matrix_root: Path,
    out_path: Path,
    x_col: str = "renders",
    x_label: str = "Renders (cumulative fitness evaluations)",
    n_seeds: int = 5,
) -> None:
    """Two panels (additive, exclusive), sharing the render-count x-axis.

    This arm — and ONLY this arm — must stay on renders. Its cells vary
    `children_ratio`, so at generation 3000 K/N=2.0 has spent 180,030 renders
    against K/N=0.5's 45,030: a 4x compute difference that a generation axis
    would silently fold into the curves. Every other figure's cells share one
    children count, which is why they can plot against generations.

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
            curves = load_seed_curves(cell_dir, x_col=x_col)
            grid_x, values = align_on_grid(curves)
            median, q1, q3 = median_iqr(values)
            ax.plot(grid_x, median, label=f"K/N={ratio}")
            ax.fill_between(grid_x, q1, q3, alpha=0.2)
        ax.set_xlabel(x_label)
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


# One hue for a single-series bar chart: the category is already carried by
# the axis, so coloring each bar by its own value would double-encode length
# as hue and burn the only free channel (anti-pattern: value-ramp on nominal
# categories). Grey is reserved here for the non-data reference line.
BAR_COLOR = "#4a6fa5"
REFERENCE_COLOR = "#8a8a8a"


def _final_value_per_seed(cell_dir: Path, y_col: str = "best_fitness") -> np.ndarray:
    """Each seed's LAST value of `y_col` for one matrix cell.

    Deliberately NOT `median_iqr(align_on_grid(...))[-1]`: aligning onto a
    shared grid truncates every seed to the shortest one's x-range, which is
    the right thing when overlaying curves and the wrong thing when the only
    question is where each seed actually finished.
    """
    curves = load_seed_curves(cell_dir, y_col=y_col)
    return np.array([y[-1] for _x, y in curves], dtype=np.float64)


def _generations_to_threshold(cell_dir: Path, threshold: float) -> np.ndarray:
    """Each seed's first generation whose best fitness reaches `threshold`.

    A seed that never reaches it contributes `np.nan`, never the run's final
    generation: "never got there" and "got there on the last generation" are
    different facts, and collapsing the first into the second would flatter
    exactly the methods that plateau below the threshold -- the ones this
    project's own results say plateau at ~0.72.
    """
    curves = load_seed_curves(cell_dir, x_col="generation", y_col="best_fitness")
    reached_at: list[float] = []
    for x, y in curves:
        hits = np.flatnonzero(y >= threshold)
        reached_at.append(float(x[hits[0]]) if hits.size else float("nan"))
    return np.array(reached_at, dtype=np.float64)


def blank_canvas_fitness(matrix_root: Path) -> float:
    """Fitness of a canvas with no triangles drawn on it at all.

    Read back from the matrix's own archived `run.json` (the target image and
    canvas size a cell actually ran with), then computed with the engine's
    real `Evaluator` on an all-zero genome: every triangle's ACTIVE gene is
    below threshold, so nothing is drawn and the frame IS the background.
    That is one render of a blank frame, not a GA run -- this script's
    never-re-run-the-GA rule is intact.

    Returned so the bar figures can baseline there instead of at zero: bars
    encode magnitude by length, and a bar chart of values that all live
    between 0.69 and 0.99 is either unreadable (zero baseline) or dishonest
    (an arbitrary crop). "What you score by doing nothing" is neither, and it
    is the same floor the presentation already argues from.
    """
    run_json = next(matrix_root.glob("*/seed*/run.json"), None)
    if run_json is None:
        raise GeneratePlotsError(
            f"no cell under {matrix_root} carried a seed*/run.json -- cannot recover the "
            "target image the bar figures baseline against"
        )
    with run_json.open(encoding="utf-8") as handle:
        config = json.load(handle)["config"]
    for key in ("image", "canvas"):
        if key not in config:
            raise GeneratePlotsError(f"{run_json} config is missing {key!r}")

    image_path = Path(config["image"])
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    canvas = int(config["canvas"])
    size = (canvas, canvas)

    target = load_target(image_path, size)
    evaluator = Evaluator(target, size)
    # Budget 1, deliberately, and NOT read from the run: `run.json` archives
    # the triangle budget nowhere (it is CLI-only, see README), and it does
    # not matter here -- an all-zero genome puts every ACTIVE gene below
    # threshold, so no triangle is drawn at any budget and every budget
    # renders the identical blank frame.
    empty = np.zeros(chromosome_length(1), dtype=np.float32)
    fitness, _frame = evaluator.evaluate(empty)
    return float(fitness)


def plot_final_bars(
    cell_dirs: dict[str, Path],
    title: str,
    out_path: Path,
    baseline: float,
    baseline_label: str,
    n_seeds: int = 5,
    annotations: dict[str, str] | None = None,
    display_names: dict[str, str] | None = None,
    value_label: str = "Best fitness final",
) -> None:
    """Horizontal bars of final best fitness, one bar per cell.

    `display_names` and `value_label` exist so the presentation build can
    relabel the SAME computation in Spanish without a second implementation
    of it -- see scripts/make_deck_figures.py. They change captions only;
    nothing downstream of the data.

    Horizontal, not vertical: these category names are long
    (`tournament_probabilistic`, `matched-multigen_uniform`) and rotated
    x-tick labels collide. Sorted by value so the ranking is the reading
    order. `baseline` sets the bar origin -- see `blank_canvas_fitness`.
    """
    means_by_label = {label: _final_value_per_seed(path) for label, path in cell_dirs.items()}
    ordered = sorted(means_by_label, key=lambda label: float(np.mean(means_by_label[label])))
    means = np.array([float(np.mean(means_by_label[k])) for k in ordered])
    stds = np.array([float(np.std(means_by_label[k])) for k in ordered])

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.52 * len(ordered) + 1.6)))
    positions = np.arange(len(ordered))
    ax.barh(
        positions, means - baseline, left=baseline, xerr=stds,
        color=BAR_COLOR, height=0.68, error_kw={"ecolor": "#333333", "capsize": 3, "lw": 1},
    )
    ax.axvline(baseline, color=REFERENCE_COLOR, lw=1.2)
    ax.set_yticks(positions)
    ax.set_yticklabels([(display_names or {}).get(label, label) for label in ordered])
    ax.set_xlabel(f"{value_label}  (barras desde {baseline_label} = {baseline:.4f})")
    upper = float(np.max(means + stds))
    span = max(upper - baseline, 1e-6)
    # Headroom for the direct labels, but never past 1.0: fitness is
    # 1 - normalized RMSE, so an axis running to 1.1 would draw a region of
    # the scale that cannot exist.
    ax.set_xlim(baseline, min(1.0, baseline + span * 1.30))
    # Direct-label every bar: with <=8 bars this is a ranking table that also
    # has a length channel, not "a number on every point" in a dense series --
    # and the differences being argued about live in the 4th decimal.
    for pos, mean, std in zip(positions, means, stds):
        text = f"{mean:.4f} +/- {std:.4f}"
        if annotations is not None:
            text += f"   [{annotations[ordered[int(pos)]]}]"
        right = ax.get_xlim()[1]
        anchor = mean + std + span * 0.02
        # A label that would run off the clamped axis flips inside the bar
        # instead of being silently cropped by bbox_inches="tight".
        if anchor + span * 0.45 > right:
            ax.text(mean - std - span * 0.02, pos, text, va="center", ha="right",
                    fontsize=8, color="#ffffff")
        else:
            ax.text(anchor, pos, text, va="center", fontsize=8, color="#333333")
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(_wrap_title(f"{title}  (n={n_seeds})", width=84))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost(
    cell_dirs: dict[str, Path],
    title: str,
    out_path: Path,
    threshold: float = 0.95,
    n_seeds: int = 5,
    display_names: dict[str, str] | None = None,
) -> None:
    """Two panels: wall-clock seconds, and generations to reach `threshold`.

    Two panels rather than two y-scales on one axes. Seconds and generations
    are different units over different ranges; overlaying them on a dual axis
    would let the scaling choice decide which method "looks" cheaper.

    Both panels baseline at zero, which is correct here and is NOT the
    compromise `plot_final_bars` has to make: a run really can take zero
    seconds and really can need zero generations, so bar length stays
    proportional to the quantity.
    """
    ordered = sorted(cell_dirs)
    seconds_raw = {k: _final_value_per_seed(cell_dirs[k], y_col="elapsed_s") for k in ordered}
    seconds = np.array([float(np.mean(seconds_raw[k])) for k in ordered])
    seconds_sd = np.array([float(np.std(seconds_raw[k])) for k in ordered])
    gens = [_generations_to_threshold(cell_dirs[k], threshold) for k in ordered]
    gens_mean = np.array([float(np.nanmean(g)) if np.any(~np.isnan(g)) else np.nan for g in gens])
    gens_sd = np.array([float(np.nanstd(g)) if np.any(~np.isnan(g)) else np.nan for g in gens])

    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.0, 0.52 * len(ordered) + 1.8)), sharey=True)
    positions = np.arange(len(ordered))

    axes[0].barh(positions, seconds, xerr=seconds_sd, color=BAR_COLOR, height=0.68,
                 error_kw={"ecolor": "#333333", "capsize": 3, "lw": 1})
    axes[0].set_xlabel("Segundos de reloj por corrida")
    axes[0].set_title("Costo en tiempo")
    seconds_cap = float(np.max(seconds + seconds_sd))
    for pos, value, err in zip(positions, seconds, seconds_sd):
        axes[0].text(value + err + seconds_cap * 0.03, pos, f"{value:.0f}s",
                     va="center", fontsize=8, color="#333333")
    axes[0].set_xlim(0, seconds_cap * 1.30)

    finite = gens_mean[~np.isnan(gens_mean)]
    reach_cap = float(np.max(finite)) if finite.size else 1.0
    axes[1].barh(positions, np.nan_to_num(gens_mean), xerr=np.nan_to_num(gens_sd),
                 color=BAR_COLOR, height=0.68, error_kw={"ecolor": "#333333", "capsize": 3, "lw": 1})
    for pos, value, err in zip(positions, gens_mean, gens_sd):
        if np.isnan(value):
            # Named, not dropped and not left as a zero-length bar with no
            # explanation: "never reached it" is the finding, not missing data.
            axes[1].text(reach_cap * 0.03, pos, f"nunca alcanza {threshold}", va="center",
                         fontsize=8, style="italic", color="#333333")
        else:
            axes[1].text(value + err + reach_cap * 0.03, pos, f"{value:.0f}",
                         va="center", fontsize=8, color="#333333")
    axes[1].set_xlabel(f"Generaciones hasta fitness {threshold}")
    axes[1].set_title(f"Costo en generaciones hasta {threshold}")
    axes[1].set_xlim(0, (reach_cap + float(np.nanmax(np.append(gens_sd, 0.0)))) * 1.22)

    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([(display_names or {}).get(label, label) for label in ordered])
    for ax in axes:
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    wrapped = _wrap_title(f"{title}  (n={n_seeds})", width=100)
    fig.tight_layout()
    fig.suptitle(wrapped, y=1.0 + 0.05 * (wrapped.count("\n") + 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mutation_faceted(
    mutation_root: Path,
    out_path: Path,
    n_seeds: int = 5,
    title: str | None = None,
    display_names: dict[str, str] | None = None,
    y_label: str = "Best fitness",
) -> None:
    """Two panels: the four scopes at equal Pm, then the same scopes rate-matched.

    Faceted rather than eight curves on one axes: past ~6 series a single
    categorical axes stops being readable, and the split IS the argument --
    panel 1 is what "compare the four mutation methods" naively means, panel
    2 is that comparison made fair. `gene` appears in both panels because it
    is the shared reference: at Pm=0.9 it already mutates 0.9 genes per
    child, so it needs no matched twin.
    """
    panels = (
        ("Mismo Pm = 0,9", ["gene", "multigen_limited", "multigen_uniform", "complete"]),
        (
            "Tasa igualada a 0,9 genes por hijo",
            ["gene", "matched-multigen_limited", "matched-multigen_uniform", "matched-complete"],
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (panel_title, labels) in zip(axes, panels):
        for label in labels:
            cell_dir = mutation_root / f"mutation-{label}"
            _require_dir(cell_dir, _mutation_matrix_missing_hint())
            curves = load_seed_curves(cell_dir, x_col="generation")
            grid_x, values = align_on_grid(curves)
            median, q1, q3 = median_iqr(values)
            expected = MUTATION_EXPECTED_GENES[label]
            shown = (display_names or {}).get(label, label)
            ax.plot(grid_x, median, label=f"{shown} (~{expected:g} genes/hijo)")
            ax.fill_between(grid_x, q1, q3, alpha=0.2)
        ax.set_xlabel("Generaciones (igual presupuesto de renders)")
        ax.set_title(panel_title)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(y_label)
    claim = title if title is not None else FIGURE_CLAIMS["fig_mutation_fitness.png"]
    wrapped = _wrap_title(f"{claim}  (n={n_seeds})", width=100)
    fig.tight_layout()
    fig.suptitle(wrapped, y=1.0 + 0.05 * (wrapped.count("\n") + 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_all_figures(
    matrix_root: Path,
    hillclimber_dir: Path,
    plots_dir: Path,
    mutation_root: Path | None = None,
) -> list[Path]:
    """Every figure the presentation cites, off `align_on_grid` + `median_iqr`.

    `mutation_root` is optional and defaults to None -- the mutation arm lives
    in its own matrix (`configs/experiments/mutation_matrix.json` ->
    `runs/matrix_mutation/`) because its cells override `mutation` rather than
    any of `main_matrix.json`'s dimensions. Passing None skips the two
    mutation figures instead of failing, so a checkout that has only ever run
    the main matrix still builds every figure it does have the data for.
    """
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

    # The controlled counterpart arm: the SAME 7 labels, applied to `parents`
    # only against a pinned elite replacement. Reuses SELECTION_LABELS rather
    # than duplicating the list, so the two arms can never drift apart.
    selection_parents_cells = {
        label: matrix_root / f"selection_parents-{label}" for label in SELECTION_LABELS
    }
    for cell_dir in selection_parents_cells.values():
        _require_dir(cell_dir, _matrix_missing_hint())

    out = plots_dir / "fig_selection_parents_fitness.png"
    plot_arm(
        selection_parents_cells, FIGURE_CLAIMS["fig_selection_parents_fitness.png"], out,
        y_col="best_fitness", y_label="Best fitness", n_seeds=n_seeds,
    )
    outputs.append(out)

    out = plots_dir / "fig_selection_parents_diversity.png"
    plot_arm(
        selection_parents_cells, FIGURE_CLAIMS["fig_selection_parents_diversity.png"], out,
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

    # The endpoint of fig_selection_fitness.png's curves, read as a ranking,
    # plus what each method cost to get there. Both re-read the same cells --
    # no extra runs.
    selection_floor = blank_canvas_fitness(matrix_root)
    out = plots_dir / "fig_selection_final_bars.png"
    plot_final_bars(
        selection_cells, FIGURE_CLAIMS["fig_selection_final_bars.png"], out,
        baseline=selection_floor, baseline_label="canvas en blanco", n_seeds=n_seeds,
    )
    outputs.append(out)

    out = plots_dir / "fig_cost.png"
    plot_cost(selection_cells, FIGURE_CLAIMS["fig_cost.png"], out, n_seeds=n_seeds)
    outputs.append(out)

    if mutation_root is not None:
        mutation_cells = {label: mutation_root / f"mutation-{label}" for label in MUTATION_LABELS}
        for cell_dir in mutation_cells.values():
            _require_dir(cell_dir, _mutation_matrix_missing_hint())
        mutation_seeds = len(load_seed_curves(next(iter(mutation_cells.values()))))

        out = plots_dir / "fig_mutation_fitness.png"
        plot_mutation_faceted(mutation_root, out, n_seeds=mutation_seeds)
        outputs.append(out)

        out = plots_dir / "fig_mutation_final_bars.png"
        plot_final_bars(
            mutation_cells, FIGURE_CLAIMS["fig_mutation_final_bars.png"], out,
            baseline=blank_canvas_fitness(mutation_root),
            baseline_label="canvas en blanco",
            n_seeds=mutation_seeds,
            # The annotation is the whole point of this figure: without the
            # genes-per-child number beside each bar, the same-Pm cells read
            # as a like-for-like operator comparison, which they are not.
            annotations={
                label: f"~{MUTATION_EXPECTED_GENES[label]:g} genes/hijo" for label in MUTATION_LABELS
            },
        )
        outputs.append(out)

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the seven named comparative figures from pre-existing matrix "
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
        "--mutation-root",
        type=Path,
        default=Path("runs/matrix_mutation"),
        help=(
            "root directory of the mutation arm's cell run directories "
            "(python -m tp2.experiments.runner --spec configs/experiments/mutation_matrix.json output). "
            "If the directory does not exist the two mutation figures are skipped, not failed"
        ),
    )
    parser.add_argument(
        "--plots-dir", type=Path, default=Path("plots"), help="output directory for generated figures"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the full seven-figure set from real matrix and hill-climber output.

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
    mutation_root = (
        args.mutation_root if args.mutation_root.is_absolute() else PROJECT_ROOT / args.mutation_root
    )
    # Absent is not an error: the mutation arm is its own matrix, and a
    # checkout that has not run it should still get every other figure. Say
    # so on stderr rather than skipping silently.
    if not mutation_root.is_dir():
        print(
            f"skipping the mutation figures: {mutation_root} does not exist -- "
            f"{_mutation_matrix_missing_hint()}",
            file=sys.stderr,
        )
        mutation_root = None

    try:
        outputs = build_all_figures(matrix_root, hillclimber_dir, plots_dir, mutation_root)
    except GeneratePlotsError as exc:
        parser.error(str(exc))
        return 2  # unreachable: argparse's .error() calls sys.exit()

    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
