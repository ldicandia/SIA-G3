"""Level file loading and parsing.

This module is the seam 01-01-SUMMARY.md predicted: the game sources its
boards from JSON level files without touching the engine or renderer.
The engine and the level layer must both stay importable with no renderer
present -- check_headless.py enforces this requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

from gridworld.engine.board import Board, Position
from gridworld.engine.state import GameState

MAX_DIMENSION = 100


class LevelProblem(Enum):
    """Specific failure modes for level loading and validation."""

    UNREADABLE_FILE = "unreadable_file"
    MALFORMED_JSON = "malformed_json"
    NOT_AN_OBJECT = "not_an_object"
    MISSING_KEY = "missing_key"
    WRONG_TYPE = "wrong_type"
    BAD_DIMENSION = "bad_dimension"
    BAD_COORDINATE = "bad_coordinate"
    OUT_OF_BOUNDS = "out_of_bounds"


class LevelError(Exception):
    """Raised when a level file cannot be read or fails validation."""

    def __init__(self, problem: LevelProblem, source: str, detail: str) -> None:
        super().__init__(f"{source}: {detail}")
        self.problem = problem
        self.source = source
        self.detail = detail


@dataclass(frozen=True, slots=True)
class Level:
    """A loaded level holding its metadata, static board, and start state."""

    name: str
    board: Board
    state: GameState


DEFAULT_LEVELS_DIR = Path(__file__).resolve().parent.parent / "levels"
DEFAULT_LEVEL_PATH = DEFAULT_LEVELS_DIR / "01-warmup.json"


def _is_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _parse_coordinate(val: Any, owner: str, source: str, rows: int, cols: int) -> Position:
    if not isinstance(val, (list, tuple)) or len(val) != 2:
        raise LevelError(
            LevelProblem.BAD_COORDINATE,
            source,
            f"{owner} coordinate must be a two-element [row, col] array, got {val!r}",
        )
    r, c = val
    if not _is_int(r) or not _is_int(c):
        raise LevelError(
            LevelProblem.BAD_COORDINATE,
            source,
            f"{owner} coordinate elements must be integers, got {val!r}",
        )
    if not (0 <= r < rows and 0 <= c < cols):
        raise LevelError(
            LevelProblem.OUT_OF_BOUNDS,
            source,
            f"{owner} coordinate [{r}, {c}] is out of bounds for {rows}x{cols} grid",
        )
    return (r, c)


def parse_level(data: object, source: str) -> Level:
    """Parse a decoded JSON object into a Level.

    Performs structural validation: required keys, data types, grid boundaries,
    and coordinate formats. Order of cars and flags is normalized by car number.
    """
    if not isinstance(data, dict):
        raise LevelError(
            LevelProblem.NOT_AN_OBJECT,
            source,
            f"level data must be a JSON object, got {type(data).__name__}",
        )

    required_keys = ["name", "rows", "cols", "obstacles", "cars", "flags"]
    for key in required_keys:
        if key not in data:
            raise LevelError(
                LevelProblem.MISSING_KEY,
                source,
                f"missing required key '{key}'",
            )

    name = data["name"]
    if not isinstance(name, str):
        raise LevelError(
            LevelProblem.WRONG_TYPE,
            source,
            f"key 'name' must be a string, got {type(name).__name__}",
        )

    rows = data["rows"]
    if not _is_int(rows):
        raise LevelError(
            LevelProblem.WRONG_TYPE,
            source,
            f"key 'rows' must be an integer, got {rows!r}",
        )
    if not (1 <= rows <= MAX_DIMENSION):
        raise LevelError(
            LevelProblem.BAD_DIMENSION,
            source,
            f"key 'rows' must be between 1 and {MAX_DIMENSION}, got {rows}",
        )

    cols = data["cols"]
    if not _is_int(cols):
        raise LevelError(
            LevelProblem.WRONG_TYPE,
            source,
            f"key 'cols' must be an integer, got {cols!r}",
        )
    if not (1 <= cols <= MAX_DIMENSION):
        raise LevelError(
            LevelProblem.BAD_DIMENSION,
            source,
            f"key 'cols' must be between 1 and {MAX_DIMENSION}, got {cols}",
        )

    obstacles_raw = data["obstacles"]
    if not isinstance(obstacles_raw, list):
        raise LevelError(
            LevelProblem.WRONG_TYPE,
            source,
            f"key 'obstacles' must be a list, got {type(obstacles_raw).__name__}",
        )

    obstacles: set[Position] = set()
    for idx, obs_raw in enumerate(obstacles_raw):
        pos = _parse_coordinate(obs_raw, f"obstacle[{idx}]", source, rows, cols)
        obstacles.add(pos)

    cars_raw = data["cars"]
    if not isinstance(cars_raw, list):
        raise LevelError(
            LevelProblem.WRONG_TYPE,
            source,
            f"key 'cars' must be a list, got {type(cars_raw).__name__}",
        )

    cars_list: list[tuple[int, Position]] = []
    for idx, car_raw in enumerate(cars_raw):
        if not isinstance(car_raw, dict):
            raise LevelError(
                LevelProblem.WRONG_TYPE,
                source,
                f"car entry[{idx}] must be an object, got {type(car_raw).__name__}",
            )
        if "number" not in car_raw or "at" not in car_raw:
            raise LevelError(
                LevelProblem.MISSING_KEY,
                source,
                f"car entry[{idx}] missing 'number' or 'at'",
            )
        num = car_raw["number"]
        if not _is_int(num) or num <= 0:
            raise LevelError(
                LevelProblem.WRONG_TYPE,
                source,
                f"car entry[{idx}] 'number' must be a positive integer, got {num!r}",
            )
        pos = _parse_coordinate(car_raw["at"], f"car {num}", source, rows, cols)
        cars_list.append((num, pos))

    flags_raw = data["flags"]
    if not isinstance(flags_raw, list):
        raise LevelError(
            LevelProblem.WRONG_TYPE,
            source,
            f"key 'flags' must be a list, got {type(flags_raw).__name__}",
        )

    flags_list: list[tuple[int, Position]] = []
    for idx, flag_raw in enumerate(flags_raw):
        if not isinstance(flag_raw, dict):
            raise LevelError(
                LevelProblem.WRONG_TYPE,
                source,
                f"flag entry[{idx}] must be an object, got {type(flag_raw).__name__}",
            )
        if "number" not in flag_raw or "at" not in flag_raw:
            raise LevelError(
                LevelProblem.MISSING_KEY,
                source,
                f"flag entry[{idx}] missing 'number' or 'at'",
            )
        num = flag_raw["number"]
        if not _is_int(num) or num <= 0:
            raise LevelError(
                LevelProblem.WRONG_TYPE,
                source,
                f"flag entry[{idx}] 'number' must be a positive integer, got {num!r}",
            )
        pos = _parse_coordinate(flag_raw["at"], f"flag {num}", source, rows, cols)
        flags_list.append((num, pos))

    cars_sorted = sorted(cars_list, key=lambda x: x[0])
    flags_sorted = sorted(flags_list, key=lambda x: x[0])

    cars_positions = tuple(pos for _, pos in cars_sorted)
    flags_positions = tuple(pos for _, pos in flags_sorted)

    board = Board(
        rows=rows,
        cols=cols,
        obstacles=frozenset(obstacles),
        flags=flags_positions,
    )
    state = GameState(
        cars=cars_positions,
        parked=frozenset(),
    )
    return Level(name=name, board=board, state=state)


def load_level(path: str | Path) -> Level:
    """Load a Level from a JSON file on disk."""
    filepath = Path(path)
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as err:
        raise LevelError(
            LevelProblem.UNREADABLE_FILE,
            str(path),
            f"cannot read level file: {err}",
        ) from err

    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as err:
        raise LevelError(
            LevelProblem.MALFORMED_JSON,
            str(path),
            f"line {err.lineno} column {err.colno} (char {err.pos})",
        ) from err

    return parse_level(decoded, str(path))
