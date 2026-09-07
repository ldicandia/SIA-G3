"""Claim-to-column audit: every figure claimed in
scripts.generate_plots.FIGURE_CLAIMS must exist as a real file on disk.

Never asserted by eye: this is the mechanical proof that T-05-04's tampering
threat (a slide figure not actually produced by the experiment runner) cannot
silently pass review.

The audit used to run in both directions, also checking that every claimed
figure was referenced from `docs/presentacion.md`. That direction died with
the deck's migration to `docs/Presentacion.pptx`: a .pptx embeds its images
rather than linking them, so there is no reference to parse and no honest way
to enforce it. Only the on-disk half survives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Runnable directly (`python scripts/audit_claims.py`), not only via
# `python -m`: put the project root on sys.path before importing
# scripts.generate_plots, the same convention scripts/generate_plots.py
# itself uses to resolve tp2 without a caller-set PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def audit(plots_dir: Path, figure_claims: dict[str, str] | None = None) -> list[str]:
    """Check every key in `figure_claims` (defaults to the real
    `scripts.generate_plots.FIGURE_CLAIMS`) against a real file under
    `plots_dir`.

    Never raises on the first problem -- collects all violations found so a
    single audit run reports everything wrong at once. Returns an empty list
    when every claimed figure exists on disk.
    """
    if figure_claims is None:
        from scripts.generate_plots import FIGURE_CLAIMS

        figure_claims = FIGURE_CLAIMS

    problems: list[str] = []
    for fig in sorted(figure_claims):
        expected_path = Path(plots_dir) / fig
        if not expected_path.is_file():
            problems.append(
                f"{fig} is a key in scripts.generate_plots.FIGURE_CLAIMS but no file "
                f"exists at {expected_path} -- missing figure on disk"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-dir", default="plots", help="Path to the directory containing the figure PNGs")
    args = parser.parse_args(argv)

    problems = audit(Path(args.plots_dir))

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1

    print(f"clean: every claimed figure exists as a real file under {args.plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
