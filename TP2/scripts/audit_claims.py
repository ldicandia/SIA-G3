"""Claim-to-column audit: every figure in docs/presentacion.md must trace to
a real scripts.generate_plots.FIGURE_CLAIMS key and a real file on disk, in
BOTH directions -- a referenced figure that is not a real claim is flagged,
and a real claim that is never referenced from the deck is also flagged.

Never asserted by eye: this is the mechanical proof that T-05-04's tampering
threat (a slide figure not actually produced by the experiment runner)
cannot silently pass review.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Runnable directly (`python scripts/audit_claims.py`), not only via
# `python -m`: put the project root on sys.path before importing
# scripts.generate_plots, the same convention scripts/generate_plots.py
# itself uses to resolve tp2 without a caller-set PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# One place figure references are parsed: a later change to the deck's
# image syntax only needs a fix here. Matches any Markdown image link whose
# target path contains a "plots/" segment and ends in ".png".
_FIGURE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)]*plots/[^)]+\.png)\)")


def extract_referenced_figures(markdown_text: str) -> set[str]:
    """Return the set of unique plots/*.png basenames referenced anywhere
    in `markdown_text`. Filename-only (via `Path(...).name`), not full
    paths, and plots-only -- an image link outside any `plots/` segment
    (e.g. `../assets/flag.png`) is not returned.
    """
    matches = _FIGURE_LINK_RE.findall(markdown_text)
    return {Path(m).name for m in matches}


def audit(markdown_path: Path, plots_dir: Path, figure_claims: dict[str, str] | None = None) -> list[str]:
    """Cross-check every figure referenced in `markdown_path` against
    `figure_claims` (defaults to the real `scripts.generate_plots.FIGURE_CLAIMS`
    when omitted) and against real files under `plots_dir`.

    Never raises on the first problem -- collects all violations found so a
    single audit run reports everything wrong at once. Returns an empty list
    when the deck is fully consistent in both directions.
    """
    if figure_claims is None:
        from scripts.generate_plots import FIGURE_CLAIMS

        figure_claims = FIGURE_CLAIMS

    markdown_text = Path(markdown_path).read_text(encoding="utf-8")
    referenced = extract_referenced_figures(markdown_text)

    problems: list[str] = []

    # (1) every referenced figure is a key in figure_claims.
    for fig in sorted(referenced):
        if fig not in figure_claims:
            problems.append(
                f"{fig} is referenced in {markdown_path} but is not a key in "
                "scripts.generate_plots.FIGURE_CLAIMS -- unclaimed figure"
            )

    # (2) every referenced figure that IS a figure_claims key has a real
    # file on disk at plots_dir / figure_name.
    for fig in sorted(referenced):
        if fig in figure_claims:
            expected_path = Path(plots_dir) / fig
            if not expected_path.is_file():
                problems.append(
                    f"{fig} is claimed and referenced but no file exists at "
                    f"{expected_path} -- missing figure on disk"
                )

    # (3) every figure_claims key is referenced by at least one image link
    # in the markdown.
    for fig in sorted(figure_claims):
        if fig not in referenced:
            problems.append(
                f"{fig} is a key in scripts.generate_plots.FIGURE_CLAIMS but is never "
                f"referenced from {markdown_path} -- claimed but unused figure"
            )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", default="docs/presentacion.md", help="Path to the slide deck Markdown file")
    parser.add_argument("--plots-dir", default="plots", help="Path to the directory containing the figure PNGs")
    args = parser.parse_args(argv)

    problems = audit(Path(args.markdown), Path(args.plots_dir))

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1

    print(f"clean: every figure referenced in {args.markdown} traces to a real claim and a real file on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
