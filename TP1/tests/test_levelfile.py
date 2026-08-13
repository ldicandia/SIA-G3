"""Tests for level file loading, validation, and execution."""

from pathlib import Path
import tempfile

import pytest

from gridworld.engine.rules import apply_move, is_solved
from gridworld.engine.state import Direction
from gridworld.levelfile import (
    DEFAULT_LEVEL_PATH,
    LevelError,
    LevelProblem,
    load_level,
    parse_level,
)


def test_warmup_level_loads_expected_board() -> None:
    level = load_level(DEFAULT_LEVEL_PATH)
    assert level.name == "Warmup"
    assert level.board.rows == 5
    assert level.board.cols == 5
    assert len(level.board.obstacles) == 2
    assert len(level.board.flags) == 2
    assert len(level.state.cars) == 2
    assert len(level.state.parked) == 0


def test_warmup_level_drives_to_solved() -> None:
    level = load_level(DEFAULT_LEVEL_PATH)
    board = level.board
    state = level.state

    moves = [
        # Car 1: down 3, right 4, down 1 -> (4, 4)
        (1, Direction.DOWN),
        (1, Direction.DOWN),
        (1, Direction.DOWN),
        (1, Direction.RIGHT),
        (1, Direction.RIGHT),
        (1, Direction.RIGHT),
        (1, Direction.RIGHT),
        (1, Direction.DOWN),
        # Car 2: down 3, left 4, down 1 -> (4, 0)
        (2, Direction.DOWN),
        (2, Direction.DOWN),
        (2, Direction.DOWN),
        (2, Direction.LEFT),
        (2, Direction.LEFT),
        (2, Direction.LEFT),
        (2, Direction.LEFT),
        (2, Direction.DOWN),
    ]

    for car, direction in moves:
        res = apply_move(board, state, car, direction)
        assert res.accepted, f"Move car {car} {direction.value} was rejected"
        state = res.state

    assert is_solved(board, state)


def test_flags_are_ordered_by_car_number() -> None:
    data = {
        "name": "Out of Order",
        "rows": 3,
        "cols": 3,
        "obstacles": [],
        "cars": [
            {"number": 2, "at": [0, 2]},
            {"number": 1, "at": [0, 0]},
        ],
        "flags": [
            {"number": 2, "at": [2, 0]},
            {"number": 1, "at": [2, 2]},
        ],
    }
    level = parse_level(data, "test")
    assert level.board.flags[0] == (2, 2)  # Car 1 flag
    assert level.board.flags[1] == (2, 0)  # Car 2 flag
    assert level.state.cars[0] == (0, 0)   # Car 1 position
    assert level.state.cars[1] == (0, 2)   # Car 2 position


def test_missing_file_raises_level_error_naming_the_path() -> None:
    missing_path = "levels/does-not-exist.json"
    with pytest.raises(LevelError) as exc_info:
        load_level(missing_path)

    err = exc_info.value
    assert err.problem == LevelProblem.UNREADABLE_FILE
    assert missing_path in str(err)


def test_malformed_json_raises_level_error_naming_the_position() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        tmp.write("{\n  \"name\": \"bad\",\n  \"rows\":\n}")
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(LevelError) as exc_info:
            load_level(tmp_path)

        err = exc_info.value
        assert err.problem == LevelProblem.MALFORMED_JSON
        assert "line" in str(err) or "column" in str(err)
    finally:
        tmp_path.unlink(missing_ok=True)
