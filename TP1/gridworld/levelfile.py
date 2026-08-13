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
    """Specific failure modes for level loading and validation.

    Structural problems:
    - UNREADABLE_FILE: level file path cannot be opened or read
    - MALFORMED_JSON: file content is not valid JSON syntax
    - NOT_AN_OBJECT: top-level JSON structure is not an object
    - MISSING_KEY: required field is missing from object
    - WRONG_TYPE: field has invalid data type
    - BAD_DIMENSION: grid dimension out of range [1, MAX_DIMENSION]
    - BAD_COORDINATE: coordinate is not a 2-element list of integers
    - OUT_OF_BOUNDS: coordinate falls outside grid dimensions

    Integrity problems:
    - NO_CARS: level declares zero cars
    - DUPLICATE_NUMBER: two cars or two flags share the same number
    - NON_CONTIGUOUS_NUMBERING: car or flag numbers are not contiguous 1..N
    - UNPAIRED_NUMBER: car without matching flag or flag without matching car
    - ON_OBSTACLE: car or flag stands on an obstacle cell
    - CELL_CONFLICT: two entities share a cell (car-car, flag-flag, car-foreign flag)
    - STARTS_ON_OWN_FLAG: car starts on its own flag
    """

    UNREADABLE_FILE = "unreadable_file"
    MALFORMED_JSON = "malformed_json"
    NOT_AN_OBJECT = "not_an_object"
    MISSING_KEY = "missing_key"
    WRONG_TYPE = "wrong_type"
    BAD_DIMENSION = "bad_dimension"
    BAD_COORDINATE = "bad_coordinate"
    OUT_OF_BOUNDS = "out_of_bounds"
    NO_CARS = "no_cars"
    DUPLICATE_NUMBER = "duplicate_number"
    NON_CONTIGUOUS_NUMBERING = "non_contiguous_numbering"
    UNPAIRED_NUMBER = "unpaired_number"
    ON_OBSTACLE = "on_obstacle"
    CELL_CONFLICT = "cell_conflict"
    STARTS_ON_OWN_FLAG = "starts_on_own_flag"


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

    Performs validation in a deterministic, fixed check order:
    1. Structural checks: top-level object, required keys, types, grid dimensions,
       and coordinate formats (out of bounds).
    2. At-least-one-car requirement (NO_CARS).
    3. Per-list numbering uniqueness (DUPLICATE_NUMBER) and contiguity 1..N
       (NON_CONTIGUOUS_NUMBERING).
    4. Pairing across car and flag lists (UNPAIRED_NUMBER).
    5. Obstacle occupancy for cars and flags (ON_OBSTACLE).
    6. Cell conflicts between entities (CELL_CONFLICT).
    7. Starts-on-own-flag prevention (STARTS_ON_OWN_FLAG).
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

    # --- Integrity Validation Matrix ---

    # 2. At-least-one-car check
    if len(cars_list) == 0:
        raise LevelError(
            LevelProblem.NO_CARS,
            source,
            "level must declare at least one car",
        )

    # 3. Per-list numbering checks
    seen_car_nums: set[int] = set()
    for num, _ in cars_list:
        if num in seen_car_nums:
            raise LevelError(
                LevelProblem.DUPLICATE_NUMBER,
                source,
                f"duplicate car number {num}",
            )
        seen_car_nums.add(num)

    seen_flag_nums: set[int] = set()
    for num, _ in flags_list:
        if num in seen_flag_nums:
            raise LevelError(
                LevelProblem.DUPLICATE_NUMBER,
                source,
                f"duplicate flag number {num}",
            )
        seen_flag_nums.add(num)

    expected_car_nums = set(range(1, len(cars_list) + 1))
    if seen_car_nums != expected_car_nums:
        missing_or_extra = sorted(seen_car_nums ^ expected_car_nums)
        raise LevelError(
            LevelProblem.NON_CONTIGUOUS_NUMBERING,
            source,
            f"car numbers must be 1 to {len(cars_list)} with no gaps, broken by number {missing_or_extra[0]} (found {sorted(seen_car_nums)})",
        )

    expected_flag_nums = set(range(1, len(flags_list) + 1))
    if seen_flag_nums != expected_flag_nums:
        missing_or_extra = sorted(seen_flag_nums ^ expected_flag_nums)
        raise LevelError(
            LevelProblem.NON_CONTIGUOUS_NUMBERING,
            source,
            f"flag numbers must be 1 to {len(flags_list)} with no gaps, broken by number {missing_or_extra[0]} (found {sorted(seen_flag_nums)})",
        )

    # 4. Pairing across the two lists
    unpaired_cars = sorted(seen_car_nums - seen_flag_nums)
    if unpaired_cars:
        num = unpaired_cars[0]
        raise LevelError(
            LevelProblem.UNPAIRED_NUMBER,
            source,
            f"car {num} has no matching flag {num}",
        )

    unpaired_flags = sorted(seen_flag_nums - seen_car_nums)
    if unpaired_flags:
        num = unpaired_flags[0]
        raise LevelError(
            LevelProblem.UNPAIRED_NUMBER,
            source,
            f"flag {num} has no matching car {num}",
        )

    cars_sorted = sorted(cars_list, key=lambda x: x[0])
    flags_sorted = sorted(flags_list, key=lambda x: x[0])

    # 5. Obstacle occupancy
    for num, pos in cars_sorted:
        if pos in obstacles:
            raise LevelError(
                LevelProblem.ON_OBSTACLE,
                source,
                f"car {num} at {list(pos)} is on an obstacle cell",
            )

    for num, pos in flags_sorted:
        if pos in obstacles:
            raise LevelError(
                LevelProblem.ON_OBSTACLE,
                source,
                f"flag {num} at {list(pos)} is on an obstacle cell",
            )

    # 6. Cell conflicts
    # Car vs Car
    for i in range(len(cars_sorted)):
        c1_num, c1_pos = cars_sorted[i]
        for j in range(i + 1, len(cars_sorted)):
            c2_num, c2_pos = cars_sorted[j]
            if c1_pos == c2_pos:
                raise LevelError(
                    LevelProblem.CELL_CONFLICT,
                    source,
                    f"car {c1_num} and car {c2_num} share cell {list(c1_pos)}",
                )

    # Flag vs Flag
    for i in range(len(flags_sorted)):
        f1_num, f1_pos = flags_sorted[i]
        for j in range(i + 1, len(flags_sorted)):
            f2_num, f2_pos = flags_sorted[j]
            if f1_pos == f2_pos:
                raise LevelError(
                    LevelProblem.CELL_CONFLICT,
                    source,
                    f"flag {f1_num} and flag {f2_num} share cell {list(f1_pos)}",
                )

    # Car vs Flag of different car
    for c_num, c_pos in cars_sorted:
        for f_num, f_pos in flags_sorted:
            if c_num != f_num and c_pos == f_pos:
                raise LevelError(
                    LevelProblem.CELL_CONFLICT,
                    source,
                    f"car {c_num} and flag {f_num} share cell {list(c_pos)}",
                )

    # 7. Starts-on-own-flag rule
    for (c_num, c_pos), (f_num, f_pos) in zip(cars_sorted, flags_sorted):
        if c_num == f_num and c_pos == f_pos:
            raise LevelError(
                LevelProblem.STARTS_ON_OWN_FLAG,
                source,
                f"car {c_num} starts on its own flag at {list(c_pos)}; level must not begin already solved",
            )

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
