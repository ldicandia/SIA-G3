"""Spanish, presentation-styled variants of the three newest figures.

`scripts/generate_plots.py` writes the repo's evidence figures: their titles
are the FIGURE_CLAIMS strings, in English, naming the claim each figure is
evidence for. That is the right caption for a figure a reader inspects
alongside the code, and the wrong one for a slide -- the deck is in Spanish
and its figures carry a short descriptive title instead.

Rather than reimplement the plots, this reuses generate_plots' own functions
with Spanish captions and operator names. Same data, same aggregation, same
honesty rules (n= on every title, the blank-canvas baseline, "nunca alcanza"
spelled out); only the words change.

    .venv/bin/python scripts/make_deck_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse  # noqa: E402

from scripts.generate_plots import (  # noqa: E402
    MUTATION_LABELS,
    SELECTION_LABELS,
    GeneratePlotsError,
    blank_canvas_fitness,
    load_seed_curves,
    plot_cost,
    plot_final_bars,
    plot_mutation_faceted,
)

# The deck already names these operators in Spanish on its earlier slides;
# these are the same names, so a reader moving between slides sees one
# vocabulary rather than the registry's internal identifiers.
SELECTION_ES = {
    "elite": "Elite",
    "roulette": "Ruleta",
    "universal": "Universal (SUS)",
    "ranking": "Ranking",
    "boltzmann": "Boltzmann",
    "tournament_deterministic": "Torneo determinístico",
    "tournament_probabilistic": "Torneo probabilístico",
}

MUTATION_ES = {
    "gene": "Gen",
    "multigen_limited": "Multigen limitada",
    "multigen_uniform": "Multigen uniforme",
    "complete": "Completa",
    "matched-multigen_limited": "Multigen limitada (igualada)",
    "matched-multigen_uniform": "Multigen uniforme (igualada)",
    "matched-complete": "Completa (igualada)",
    "gene-non_uniform": "Gen, calendario no uniforme",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=PROJECT_ROOT / "runs/matrix")
    parser.add_argument("--mutation-root", type=Path, default=PROJECT_ROOT / "runs/matrix_mutation")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "plots/deck")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        selection_cells = {label: args.matrix_root / f"selection-{label}" for label in SELECTION_LABELS}
        n_seeds = len(load_seed_curves(selection_cells["elite"]))
        floor = blank_canvas_fitness(args.matrix_root)

        out = args.out_dir / "fig_selection_final_bars.png"
        plot_final_bars(
            selection_cells,
            "Fitness final por método de selección",
            out,
            baseline=floor,
            baseline_label="canvas en blanco",
            n_seeds=n_seeds,
            display_names=SELECTION_ES,
            value_label="Mejor fitness final",
        )
        print(out)

        out = args.out_dir / "fig_cost.png"
        plot_cost(
            selection_cells,
            "Costo por método de selección: reloj y generaciones hasta 0,95",
            out,
            n_seeds=n_seeds,
            display_names=SELECTION_ES,
        )
        print(out)

        mutation_cells = {label: args.mutation_root / f"mutation-{label}" for label in MUTATION_LABELS}
        mutation_seeds = len(load_seed_curves(mutation_cells["gene"]))

        out = args.out_dir / "fig_mutation_fitness.png"
        plot_mutation_faceted(
            args.mutation_root,
            out,
            n_seeds=mutation_seeds,
            title="Métodos de mutación — el mismo Pm no es la misma tasa de mutación",
            display_names=MUTATION_ES,
            y_label="Mejor fitness",
        )
        print(out)
    except GeneratePlotsError as exc:
        parser.error(str(exc))
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
