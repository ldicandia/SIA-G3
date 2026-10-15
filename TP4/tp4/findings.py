"""Evidence checker for findings notes (`docs/hallazgos/*.md`).

    python -m tp4.findings docs/hallazgos/ej1.md --cite runs/pca/summary.json:pc1_explained_variance_pct

Checks that (1) every `runs/...` artifact the note references exists and
(2) every cited summary value appears in the note text (decimal point or
decimal comma). Exit code 0 when everything is present, 1 otherwise.

Standard library only: no plotting, no imports from tp4.experiments.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ARTIFACT_RE = re.compile(r"runs/[A-Za-z0-9_./-]+\.(?:png|csv|json)")
_MISSING = object()


def referenced_artifacts(text: str) -> list[str]:
    """Sorted unique `runs/...` artifact paths mentioned in `text` (links or plain text)."""
    return sorted(set(ARTIFACT_RE.findall(text)))


def cited_value_present(text: str, value: Any) -> bool:
    """Does `value` appear in `text`?

    bool -> "true"/"false" (case-insensitive); int -> digits not glued to other digits
    or a decimal part; float -> `repr(value)` with decimal point or comma; str -> verbatim;
    list -> every element present.
    """
    if isinstance(value, bool):
        return str(value).lower() in text.lower()
    if isinstance(value, int):
        return re.search(rf"(?<![\d.,]){re.escape(str(value))}(?!\d|[.,]\d)", text) is not None
    if isinstance(value, float):
        literal = repr(value)
        return literal in text or literal.replace(".", ",") in text
    if isinstance(value, str):
        return value in text
    if isinstance(value, list):
        return all(cited_value_present(text, item) for item in value)
    return False


def _lookup(data: Any, dotted: str) -> Any:
    node = data
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return _MISSING
    return node


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tp4.findings", description="Check that a findings note is backed by runs/ artifacts."
    )
    parser.add_argument("md", type=Path, help="markdown note to check")
    parser.add_argument(
        "--cite",
        action="append",
        default=[],
        metavar="FILE:KEY",
        help="JSON file (relative to --root) and dotted key whose value must appear in the note",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)

    text = args.md.read_text(encoding="utf-8")
    problems = 0

    artifacts = referenced_artifacts(text)
    for artifact in artifacts:
        if not (args.root / artifact).is_file():
            print(f"MISSING artifact: {artifact}")
            problems += 1

    for spec in args.cite:
        file, _, key = spec.rpartition(":")
        if not file or not key:
            print(f"MISSING cite: {spec} (expected FILE:KEY)")
            problems += 1
            continue
        try:
            data = json.loads((args.root / file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"MISSING cite: {file}:{key} (cannot read {file})")
            problems += 1
            continue
        value = _lookup(data, key)
        if value is _MISSING or value is None:
            print(f"MISSING cite: {file}:{key} (key not found or null)")
            problems += 1
        elif not cited_value_present(text, value):
            print(f"MISSING cite: {file}:{key}={value}")
            problems += 1

    if problems:
        return 1
    print(f"ok: {len(artifacts)} artifacts, {len(args.cite)} cites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
