"""Prove the engine and level layer import and run with no rendering library present.

Decision D-03 makes this script the **sole mechanism** for the ENG-08
guarantee. A complementary fast pytest guard -- asserting the rendering
library stays out of ``sys.modules`` -- was offered during discussion and
explicitly declined. The accepted consequence is that a rendering-library
import introduced into the engine mid-phase is not caught until this
script is next run, so it **must be run at phase verification and again
at Phase 4 delivery**.

Run it with any Python 3 interpreter::

    py -3 TP1/scripts/check_headless.py

The script never imports ``gridworld`` into its own interpreter -- it is
the harness, and importing the package here would prove nothing about the
child environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TP1_DIR = SCRIPT_DIR.parent
REPO_ROOT = TP1_DIR.parent
VENV_DIR = TP1_DIR / ".venv-headless"
GITIGNORE_ENTRY = "TP1/.venv-headless/"

RENDERING_LIBRARY = "pygame"


def _venv_python(venv_dir: Path) -> Path:
    """Return the child interpreter path for this platform."""
    candidate = venv_dir / "Scripts" / "python.exe"
    if candidate.exists():
        return candidate
    return venv_dir / "bin" / "python"


def build_clean_venv() -> Path:
    """Create ``TP1/.venv-headless`` with pip and nothing else installed.

    The path is resolved from this file's own location, never from an
    argument and never from the working directory, so it cannot be
    redirected by a caller.
    """
    resolved = VENV_DIR.resolve()

    if resolved.name != ".venv-headless":
        raise SystemExit(f"refusing to touch unexpected path: {resolved}")
    if resolved.parent != TP1_DIR.resolve():
        raise SystemExit(f"refusing to touch path outside TP1: {resolved}")

    builder = venv.EnvBuilder(with_pip=True, clear=True)
    builder.create(str(resolved))

    _ensure_gitignored()
    return resolved


def _ensure_gitignored() -> None:
    gitignore = REPO_ROOT / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    if GITIGNORE_ENTRY in existing:
        return
    separator = "" if existing.endswith("\n") or not existing else "\n"
    gitignore.write_text(
        f"{existing}{separator}{GITIGNORE_ENTRY}\n", encoding="utf-8"
    )


def run_in_clean_venv(venv_dir: Path, code: str) -> subprocess.CompletedProcess:
    """Execute ``code`` in the clean venv's own interpreter.

    The engine is reached through ``PYTHONPATH``; nothing is installed.
    The argument list is a real list and no shell is involved.
    """
    python = _venv_python(venv_dir)
    if not python.exists():
        raise SystemExit(f"clean venv interpreter missing: {python}")

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(TP1_DIR)

    return subprocess.run(
        [str(python), "-"],
        input=code,
        capture_output=True,
        text=True,
        env=child_env,
    )


CHECK_CLEAN = f"""
try:
    import {RENDERING_LIBRARY}
except ImportError:
    print("the rendering library is genuinely absent")
else:
    raise SystemExit(
        "CONTAMINATED: {RENDERING_LIBRARY} is importable in the clean "
        "environment, so every later check would pass vacuously"
    )
"""

CHECK_IMPORTS = """
import gridworld.engine
import gridworld.engine.board
import gridworld.engine.state
import gridworld.engine.rules
import gridworld.levels
import gridworld.levelfile
import sys
assert "pygame" not in sys.modules, "importing the engine pulled in the renderer"
print("engine and level modules import with no renderer in sys.modules")
"""

CHECK_RUNS = """
from gridworld.engine.rules import apply_move, is_solved
from gridworld.engine.state import Direction, GameState
from gridworld.levels import built_in_level

board, state = built_in_level()
assert not is_solved(board, state)

sequence = (
    [(1, Direction.DOWN)] * 5
    + [(1, Direction.RIGHT)] * 6
    + [(1, Direction.DOWN)]
    + [(3, Direction.UP)] * 6
    + [(3, Direction.RIGHT)] * 3
    + [(2, Direction.DOWN)] * 5
    + [(2, Direction.LEFT)] * 3
    + [(2, Direction.DOWN)]
)

moves = 0
for car, direction in sequence:
    result = apply_move(board, state, car, direction)
    assert result.accepted, (car, direction, result.rejection)
    state = result.state
    moves += 1

assert is_solved(board, state), "the built-in level did not reach solved"
assert moves == 30, moves

visited = {state: "solved"}
twin = GameState(cars=state.cars, parked=state.parked)
assert visited[twin] == "solved", "state failed as a dictionary key"
assert len({state, twin}) == 1, "equal states did not collapse in a set"

import sys
assert "pygame" not in sys.modules, "running the engine pulled in the renderer"
print("drove the built-in level to solved in", moves, "moves; states work as dict keys")
"""

CHECK_LEVEL_FILE = """
from gridworld.engine.rules import apply_move, is_solved
from gridworld.engine.state import Direction
from gridworld.levelfile import load_level, DEFAULT_LEVEL_PATH, LevelError, LevelProblem
import sys

level = load_level(DEFAULT_LEVEL_PATH)
board, state = level.board, level.state
assert not is_solved(board, state)

sequence = (
    [(1, Direction.DOWN)] * 3
    + [(1, Direction.RIGHT)] * 4
    + [(1, Direction.DOWN)]
    + [(2, Direction.DOWN)] * 3
    + [(2, Direction.LEFT)] * 4
    + [(2, Direction.DOWN)]
)

moves = 0
for car, direction in sequence:
    result = apply_move(board, state, car, direction)
    assert result.accepted, (car, direction, result.rejection)
    state = result.state
    moves += 1

assert is_solved(board, state), "warmup level did not reach solved"
assert moves == 16, moves
assert "pygame" not in sys.modules, "loading levels pulled in the renderer"

missing_path = "levels/does-not-exist.json"
try:
    load_level(missing_path)
except LevelError as err:
    assert err.problem == LevelProblem.UNREADABLE_FILE
    assert missing_path in str(err)
else:
    raise AssertionError("missing file did not raise LevelError")

print("drove warmup level file to solved in 16 moves; error handling verified headlessly")
"""


CHECK_LEGAL_MOVES = """
from gridworld.engine.rules import LegalMove, apply_move, legal_moves, legal_moves_for
from gridworld.engine.state import Direction, GameState
from gridworld.levelfile import load_level, DEFAULT_LEVEL_PATH
import sys

level = load_level(DEFAULT_LEVEL_PATH)
board, state = level.board, level.state

expected = (
    LegalMove(car=1, direction=Direction.DOWN, target=(1, 0)),
    LegalMove(car=1, direction=Direction.RIGHT, target=(0, 1)),
    LegalMove(car=2, direction=Direction.DOWN, target=(1, 4)),
    LegalMove(car=2, direction=Direction.LEFT, target=(0, 3)),
)
moves = legal_moves(board, state)
assert moves == expected, moves

enumerated = {(m.car, m.direction) for m in moves}
for car in board.car_numbers():
    for direction in Direction:
        result = apply_move(board, state, car, direction)
        pair = (car, direction)
        assert result.accepted == (pair in enumerated), (pair, result.accepted)

all_parked = GameState(cars=state.cars, parked=frozenset(board.car_numbers()))
assert legal_moves(board, all_parked) == ()
assert legal_moves_for(board, all_parked, 1) == ()

assert "pygame" not in sys.modules, "enumerating legal moves pulled in the renderer"
print("enumerated 4 legal moves for the warmup level; agrees with apply_move; all-parked enumerates empty")
"""


def _report(label: str, result: subprocess.CompletedProcess) -> bool:
    if result.returncode == 0:
        print(f"PASS  {label}: {result.stdout.strip()}")
        return True
    print(f"FAIL  {label}")
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip())
    return False


def main() -> int:
    print(f"harness interpreter: {sys.executable}")
    print(f"clean environment:   {VENV_DIR}")

    venv_dir = build_clean_venv()

    checks = (
        ("environment is genuinely clean", CHECK_CLEAN),
        ("engine imports headlessly", CHECK_IMPORTS),
        ("engine runs headlessly", CHECK_RUNS),
        ("level file layer runs headlessly", CHECK_LEVEL_FILE),
        ("legal-move enumeration runs headlessly", CHECK_LEGAL_MOVES),
    )

    for label, code in checks:
        result = run_in_clean_venv(venv_dir, code)
        if not _report(label, result):
            return 1

    print("\nENG-08 holds: the engine, level layer, and legal-move enumeration import, run")
    print("full solutions, and use states as dictionary keys in an environment with no")
    print("renderer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

