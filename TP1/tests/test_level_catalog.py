"""Catalogue tests for all shipped JSON level files.

Proves that every shipped level file is discoverable, loads cleanly, has a valid
recorded winning solution, strictly increases in complexity across tiers, and
contains no parking-order traps.
"""

from __future__ import annotations

from pathlib import Path

from gridworld.engine.rules import apply_move, is_solved
from gridworld.engine.state import Direction
from gridworld.levels import built_in_level
from gridworld.levelfile import DEFAULT_LEVELS_DIR, load_level


SOLUTIONS: dict[str, list[tuple[int, Direction]]] = {
    "01-warmup.json": (
        [(1, Direction.DOWN)] * 3
        + [(1, Direction.RIGHT)] * 4
        + [(1, Direction.DOWN)]
        + [(2, Direction.DOWN)] * 3
        + [(2, Direction.LEFT)] * 4
        + [(2, Direction.DOWN)]
    ),
    "02-classic.json": (
        [(1, Direction.DOWN)] * 5
        + [(1, Direction.RIGHT)] * 6
        + [(1, Direction.DOWN)]
        + [(3, Direction.UP)] * 6
        + [(3, Direction.RIGHT)] * 3
        + [(2, Direction.DOWN)] * 5
        + [(2, Direction.LEFT)] * 3
        + [(2, Direction.DOWN)]
    ),
    "03-gridlock.json": (
        [(1, Direction.DOWN)] * 3
        + [(1, Direction.RIGHT)] * 1
        + [(1, Direction.DOWN)] * 2
        + [(1, Direction.LEFT)] * 1
        + [(1, Direction.DOWN)] * 3
        + [(2, Direction.DOWN)] * 3
        + [(2, Direction.RIGHT)] * 1
        + [(2, Direction.DOWN)] * 2
        + [(2, Direction.LEFT)] * 1
        + [(2, Direction.DOWN)] * 3
        + [(3, Direction.DOWN)] * 3
        + [(3, Direction.LEFT)] * 1
        + [(3, Direction.DOWN)] * 2
        + [(3, Direction.RIGHT)] * 1
        + [(3, Direction.DOWN)] * 3
        + [(4, Direction.DOWN)] * 3
        + [(4, Direction.LEFT)] * 1
        + [(4, Direction.DOWN)] * 2
        + [(4, Direction.RIGHT)] * 1
        + [(4, Direction.DOWN)] * 3
    ),
}


def _get_shipped_level_files() -> list[Path]:
    return sorted(DEFAULT_LEVELS_DIR.glob("*.json"))


def test_shipped_levels_are_discoverable():
    files = _get_shipped_level_files()
    assert len(files) >= 3, f"Expected at least 3 level files, found {len(files)}"
    for f in files:
        assert f.name in SOLUTIONS, f"Level file {f.name} has no recorded solution in SOLUTIONS"


def test_every_shipped_level_loads():
    files = _get_shipped_level_files()
    for f in files:
        level = load_level(f)
        num_cars = len(level.board.flags)
        assert num_cars >= 1, f"Level {f.name} has 0 cars"
        assert len(level.state.cars) == num_cars, (
            f"Level {f.name} has {len(level.board.flags)} flags for {len(level.state.cars)} cars"
        )


def test_every_shipped_level_is_winnable():
    files = _get_shipped_level_files()
    for f in files:
        level = load_level(f)
        board, state = level.board, level.state
        assert not is_solved(board, state), f"Level {f.name} begins already solved"

        solution = SOLUTIONS[f.name]
        for idx, (car, direction) in enumerate(solution):
            result = apply_move(board, state, car, direction)
            assert result.accepted, (
                f"Level '{f.name}' move {idx} (car {car}, {direction.name}) "
                f"rejected: {result.rejection}"
            )
            state = result.state

        assert is_solved(board, state), f"Level '{f.name}' not solved after executing solution"


def test_complexity_increases_across_tiers():
    files = _get_shipped_level_files()
    levels = [load_level(f) for f in files]

    grid_areas = [lvl.board.rows * lvl.board.cols for lvl in levels]
    car_counts = [len(lvl.board.flags) for lvl in levels]
    obstacle_counts = [len(lvl.board.obstacles) for lvl in levels]

    for i in range(len(levels) - 1):
        assert grid_areas[i] < grid_areas[i + 1], (
            f"Grid area not strictly increasing between {files[i].name} ({grid_areas[i]}) "
            f"and {files[i+1].name} ({grid_areas[i+1]})"
        )
        assert car_counts[i] < car_counts[i + 1], (
            f"Car count not strictly increasing between {files[i].name} ({car_counts[i]}) "
            f"and {files[i+1].name} ({car_counts[i+1]})"
        )
        assert obstacle_counts[i] < obstacle_counts[i + 1], (
            f"Obstacle count not strictly increasing between {files[i].name} ({obstacle_counts[i]}) "
            f"and {files[i+1].name} ({obstacle_counts[i+1]})"
        )


def test_classic_level_matches_the_code_defined_board():
    """Cross-check that keeps the file path and code path cell-for-cell identical.

    The code-defined board (built_in_level) is what Phase 1 tests and the headless
    guard run against; 02-classic.json must match it value for value.
    """
    level = load_level(DEFAULT_LEVELS_DIR / "02-classic.json")
    code_board, code_state = built_in_level()
    assert level.board == code_board
    assert level.state == code_state


def test_no_level_requires_a_lucky_parking_order():
    """Replay recorded solutions and assert no move enters a cell with a parked car or foreign flag."""
    files = _get_shipped_level_files()
    for f in files:
        level = load_level(f)
        board, state = level.board, level.state
        solution = SOLUTIONS[f.name]

        for idx, (car, direction) in enumerate(solution):
            curr_pos = state.position_of(car)
            dr, dc = direction.delta
            target_pos = (curr_pos[0] + dr, curr_pos[1] + dc)

            # Check that target_pos is not occupied by an already-parked car
            for parked_car in state.parked:
                parked_pos = state.position_of(parked_car)
                assert target_pos != parked_pos, (
                    f"Level '{f.name}' move {idx} (car {car}) enters cell {target_pos} "
                    f"occupied by already-parked car {parked_car}"
                )

            # Check that target_pos is not a foreign flag
            car_flag = board.flag_for(car)
            for flag_num, flag_pos in enumerate(board.flags, start=1):
                if flag_num != car and flag_pos != car_flag:
                    assert target_pos != flag_pos, (
                        f"Level '{f.name}' move {idx} (car {car}) enters cell {target_pos} "
                        f"holding foreign flag {flag_num}"
                    )

            result = apply_move(board, state, car, direction)
            assert result.accepted
            state = result.state
