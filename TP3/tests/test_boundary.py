"""The C++/Python boundary is the binary and its JSON — guarded as tests.

- core/ (and the CMake build) must not depend on Python, a plotting library,
  a linear-algebra library, or any entropy source other than --seed.
- scripts/ and tests/ must never import or link the core in-process.
- Only scripts/ may import the plotting library.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE = REPO_ROOT / "core"
SCRIPTS = REPO_ROOT / "scripts"
TESTS = REPO_ROOT / "tests"

PLOTTING_MODULE = "matplotlib"

# Substrings that must never appear in the C++ side.
CORE_FORBIDDEN = [
    "Python.h",  # CPython C-API header
    "pybind11",  # in-process binding library
    "Eigen",  # linear-algebra library
    PLOTTING_MODULE,
    "random_device",  # non-seed entropy source
]

# Modules that would bridge Python into the C++ code in-process.
BRIDGE_MODULES = ("ctypes", "cffi", "cppyy", "pybind11")
IMPORT_LINE = re.compile(r"^\s*(import|from)\s+(" + "|".join(BRIDGE_MODULES) + r")\b")
PLOT_IMPORT_LINE = re.compile(r"^\s*(import|from)\s+" + PLOTTING_MODULE + r"\b")
DYNAMIC_LOAD = "CDLL" + "("


def _core_files() -> list[Path]:
    files = sorted(CORE.rglob("*.hpp")) + sorted(CORE.rglob("*.cpp"))
    files.append(REPO_ROOT / "CMakeLists.txt")
    return files


def _python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def test_core_has_no_python_plotting_or_linear_algebra_dependency() -> None:
    files = _core_files()
    assert files, "core/ sources not found"
    offenders = [
        (path.relative_to(REPO_ROOT), needle)
        for path in files
        for needle in CORE_FORBIDDEN
        if needle in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_python_side_never_imports_the_core() -> None:
    files = _python_files(SCRIPTS) + _python_files(TESTS)
    assert files, "no Python files found"
    offenders = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if DYNAMIC_LOAD in text:
            offenders.append((path.relative_to(REPO_ROOT), DYNAMIC_LOAD))
        for lineno, line in enumerate(text.splitlines(), 1):
            if IMPORT_LINE.match(line):
                offenders.append((path.relative_to(REPO_ROOT), lineno, line.strip()))
    assert offenders == []


def test_only_scripts_import_the_plotting_module() -> None:
    def importers(root: Path) -> list[Path]:
        return [
            path.relative_to(REPO_ROOT)
            for path in _python_files(root)
            if any(PLOT_IMPORT_LINE.match(line) for line in path.read_text(encoding="utf-8").splitlines())
        ]

    assert importers(TESTS) == []
    assert importers(SCRIPTS) == [Path("scripts") / "validation" / "plot_validation.py"]
