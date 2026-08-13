"""Tests for level file loading, validation, and execution."""

from pathlib import Path
import tempfile
from typing import Any

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


# --- Plan 02-02 Integrity Rejection Tests ---


def test_rejects_level_with_no_cars(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["cars"] = []
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.NO_CARS
    assert "at least one car" in str(err)


def test_rejects_zero_dimension(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["rows"] = 0
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.BAD_DIMENSION
    assert "rows" in str(err)
    assert "0" in str(err)


def test_rejects_negative_dimension(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["cols"] = -3
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.BAD_DIMENSION
    assert "cols" in str(err)
    assert "-3" in str(err)


def test_rejects_non_integer_dimension(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["rows"] = 5.5
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.WRONG_TYPE
    assert "rows" in str(err)
    assert "5.5" in str(err)


def test_rejects_boolean_dimension(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["rows"] = True
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.WRONG_TYPE
    assert "rows" in str(err)
    assert "True" in str(err)


def test_rejects_dimension_above_maximum(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["cols"] = 101
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.BAD_DIMENSION
    assert "cols" in str(err)
    assert "101" in str(err)


def test_rejects_missing_required_key(valid_level_dict: dict[str, Any]) -> None:
    del valid_level_dict["name"]
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.MISSING_KEY
    assert "name" in str(err)


def test_rejects_wrong_type_for_key(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["name"] = 123
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.WRONG_TYPE
    assert "name" in str(err)


def test_rejects_top_level_not_an_object() -> None:
    with pytest.raises(LevelError) as exc_info:
        parse_level(["not", "an", "object"], "test")

    err = exc_info.value
    assert err.problem == LevelProblem.NOT_AN_OBJECT
    assert "list" in str(err)


def test_rejects_malformed_coordinate(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["obstacles"] = [[1, 2, 3]]
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.BAD_COORDINATE
    assert "obstacle[0]" in str(err)


def test_rejects_coordinate_outside_grid(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["obstacles"] = [[9, 0]]
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.OUT_OF_BOUNDS
    assert "[9, 0]" in str(err)
    assert "3x3" in str(err)


def test_rejects_duplicate_car_number(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["cars"] = [
        {"number": 1, "at": [0, 0]},
        {"number": 1, "at": [0, 2]},
    ]
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.DUPLICATE_NUMBER
    assert "1" in str(err)


def test_rejects_duplicate_flag_number(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["flags"] = [
        {"number": 2, "at": [2, 2]},
        {"number": 2, "at": [2, 0]},
    ]
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.DUPLICATE_NUMBER
    assert "2" in str(err)


def test_rejects_non_contiguous_car_numbers(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["cars"] = [
        {"number": 1, "at": [0, 0]},
        {"number": 3, "at": [0, 2]},
    ]
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.NON_CONTIGUOUS_NUMBERING
    assert "2" in str(err)


def test_rejects_car_without_matching_flag(valid_level_dict: dict[str, Any], write_level: Any) -> None:
    valid_level_dict["cars"].append({"number": 3, "at": [0, 1]})
    lvl_path = write_level(valid_level_dict)
    with pytest.raises(LevelError) as exc_info:
        load_level(lvl_path)

    err = exc_info.value
    assert err.problem == LevelProblem.UNPAIRED_NUMBER
    assert "car 3" in str(err)


def test_rejects_flag_without_matching_car(valid_level_dict: dict[str, Any], write_level: Any) -> None:
    valid_level_dict["flags"].append({"number": 3, "at": [2, 1]})
    lvl_path = write_level(valid_level_dict)
    with pytest.raises(LevelError) as exc_info:
        load_level(lvl_path)

    err = exc_info.value
    assert err.problem == LevelProblem.UNPAIRED_NUMBER
    assert "flag 3" in str(err)


def test_rejects_car_on_obstacle(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["cars"][0]["at"] = [1, 1]
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.ON_OBSTACLE
    assert "car 1" in str(err)
    assert "[1, 1]" in str(err)


def test_rejects_flag_on_obstacle(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["flags"][0]["at"] = [1, 1]
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.ON_OBSTACLE
    assert "flag 1" in str(err)
    assert "[1, 1]" in str(err)


def test_rejects_two_cars_on_one_cell(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["cars"][1]["at"] = [0, 0]
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.CELL_CONFLICT
    assert "car 1" in str(err)
    assert "car 2" in str(err)
    assert "[0, 0]" in str(err)


def test_rejects_car_on_another_cars_flag(valid_level_dict: dict[str, Any]) -> None:
    valid_level_dict["cars"][0]["at"] = [2, 0]  # Car 1 placed on Flag 2's cell
    with pytest.raises(LevelError) as exc_info:
        parse_level(valid_level_dict, "test")

    err = exc_info.value
    assert err.problem == LevelProblem.CELL_CONFLICT
    assert "car 1" in str(err)
    assert "flag 2" in str(err)
    assert "[2, 0]" in str(err)


def test_rejects_car_starting_on_own_flag(valid_level_dict: dict[str, Any], write_level: Any) -> None:
    valid_level_dict["cars"][0]["at"] = [2, 2]  # Car 1 placed on Flag 1's cell
    lvl_path = write_level(valid_level_dict)
    with pytest.raises(LevelError) as exc_info:
        load_level(lvl_path)

    err = exc_info.value
    assert err.problem == LevelProblem.STARTS_ON_OWN_FLAG
    assert "car 1" in str(err)
    assert "[2, 2]" in str(err)


def test_valid_level_dict_parses_without_error(valid_level_dict: dict[str, Any]) -> None:
    level = parse_level(valid_level_dict, "test")
    assert level.name == "Fixture Level"
    assert level.board.rows == 3
    assert level.board.cols == 3
    assert len(level.state.cars) == 2
