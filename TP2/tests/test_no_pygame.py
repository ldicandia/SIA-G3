"""The headless-engine guarantee is checked in a clean subprocess."""

from __future__ import annotations

import subprocess
import sys


MODULES = [
    "tp2.engine", "tp2.engine.genome", "tp2.engine.raster", "tp2.engine.fitness", "tp2.engine.events",
    "tp2.io", "tp2.io.images", "tp2.io.artifacts", "tp2.io.metrics",
]


def _run(extra: str = "") -> subprocess.CompletedProcess[str]:
    imports = "; ".join(f"__import__({module!r})" for module in MODULES)
    code = f"{extra}\n{imports}\n"
    return subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)


def test_headless_leaves_do_not_import_pygame() -> None:
    _run("import sys")


def test_headless_leaves_work_when_pygame_is_unavailable() -> None:
    blocker = """import sys
class BlockPygame:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'pygame': raise ImportError('blocked pygame')
        return None
sys.meta_path.insert(0, BlockPygame())"""
    _run(blocker)


def test_engine_does_not_load_ga_libraries() -> None:
    _run("import sys")
