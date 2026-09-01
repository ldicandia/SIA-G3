"""Structural, static proof that `tp2/ui/viewer.py` is the only pygame
importer in the repository (04-01 Task 2).

Two independent layers, matching ARCHITECTURE.md's "three layers" list:

1. A subprocess import test (like `tests/test_no_pygame.py`, but explicitly
   excluding `tp2.ui`): proves a headless invocation never loads pygame.
2. A static source scan: proves no OTHER file under `tp2/` contains the
   literal string `import pygame`, immune to which leaf modules a given test
   run happens to import (catches an indirect "just for the type" leak that
   an import-based test alone would miss).

Deliberately does not modify `tests/test_no_pygame.py` -- that file is
Phase 1's own guard and stays exactly as it left it.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Every leaf module Phase 1/2/3 shipped under tp2.engine and tp2.io, plus the
# composition root -- deliberately NOT tp2.ui.
MODULES = [
    "tp2.cli",
    "tp2.engine",
    "tp2.engine.config",
    "tp2.engine.diversity",
    "tp2.engine.events",
    "tp2.engine.fitness",
    "tp2.engine.genome",
    "tp2.engine.loop",
    "tp2.engine.raster",
    "tp2.engine.stop",
    "tp2.engine.operators",
    "tp2.engine.operators.crossover",
    "tp2.engine.operators.mutation",
    "tp2.engine.operators.registry",
    "tp2.engine.operators.sampling",
    "tp2.engine.operators.selection",
    "tp2.engine.operators.survival",
    "tp2.io",
    "tp2.io.artifacts",
    "tp2.io.images",
    "tp2.io.metrics",
]


def test_engine_and_cli_leaves_never_load_pygame_when_ui_is_never_imported() -> None:
    imports = "; ".join(f"__import__({module!r})" for module in MODULES)
    code = (
        "import sys\n"
        f"{imports}\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'pygame')\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_import_pygame_appears_in_exactly_one_file_under_tp2() -> None:
    tp2_root = PROJECT_ROOT / "tp2"
    offenders = {
        path.relative_to(PROJECT_ROOT)
        for path in tp2_root.rglob("*.py")
        if any(line.strip() == "import pygame" for line in path.read_text(encoding="utf-8").splitlines())
    }
    assert offenders == {pathlib.Path("tp2/ui/viewer.py")}, offenders
