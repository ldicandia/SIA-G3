"""Shared fixtures for the engine test suite.

Per decision D-04 this suite covers the engine only. Nothing here imports
the renderer package or constructs a display surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gridworld.engine.board import Board
from gridworld.engine.state import Direction, GameState
from gridworld.levels import built_in_level


@pytest.fixture
def built_in() -> tuple[Board, GameState]:
    """The built-in 7x7 three-car level: ``(board, start_state)``."""
    return built_in_level()


@pytest.fixture
def solution_sequence() -> list[tuple[int, Direction]]:
    """The 30-move winning sequence for the built-in board.

    Car 1 down five, right six, down one; car 3 up six, right three;
    car 2 down five, left three, down one.
    """
    return (
        [(1, Direction.DOWN)] * 5
        + [(1, Direction.RIGHT)] * 6
        + [(1, Direction.DOWN)]
        + [(3, Direction.UP)] * 6
        + [(3, Direction.RIGHT)] * 3
        + [(2, Direction.DOWN)] * 5
        + [(2, Direction.LEFT)] * 3
        + [(2, Direction.DOWN)]
    )


@pytest.fixture
def tiny_board() -> tuple[Board, GameState]:
    """A 3x3 board where every rejection reason is one move from the start.

    Layout, ``(row, col)`` with row 0 at the top::

        col:      0        1        2
        row 0:  car 1    car 2    flag 2
        row 1:  flag 1   obstacle    .
        row 2:    .         .        .

    Entities:

    - car 1 starts at ``(0, 0)``, its flag is at ``(1, 0)``
    - car 2 starts at ``(0, 1)``, its flag is at ``(0, 2)``
    - one obstacle at ``(1, 1)``

    From the start position car 2 at ``(0, 1)`` has all four rejection
    reasons as immediate neighbours:

    - UP    -> ``(-1, 1)``  off the grid          -> ``OFF_GRID``
    - DOWN  -> ``(1, 1)``   the obstacle          -> ``OBSTACLE``
    - LEFT  -> ``(0, 0)``   car 1 sitting there   -> ``OCCUPIED``
    - RIGHT -> ``(0, 2)``   car 2's own flag      -> accepted, parks

    and car 1 at ``(0, 0)`` supplies the foreign-flag case:

    - DOWN  -> ``(1, 0)``   car 1's own flag      -> accepted, parks
    - RIGHT -> ``(0, 1)``   car 2 sitting there   -> ``OCCUPIED``

    Car 2 moving LEFT twice would reach ``(0, 0)``; instead the
    foreign-flag case is exercised by moving car 2 onto car 1's flag at
    ``(1, 0)`` once car 1 has vacated, and by car 1 attempting to enter
    car 2's flag at ``(0, 2)``.
    """
    board = Board(
        rows=3,
        cols=3,
        obstacles=frozenset({(1, 1)}),
        flags=((1, 0), (0, 2)),
    )
    state = GameState(cars=((0, 0), (0, 1)), parked=frozenset())
    return board, state


@pytest.fixture
def stranded_board() -> tuple[Board, GameState]:
    """A 3x3 board where car 1 is boxed in and car 2 is not.

    Layout, ``(row, col)`` with row 0 at the top::

        col:      0          1          2
        row 0:  car 1     obstacle       .
        row 1: obstacle      .           .
        row 2:  car 2      flag 2     flag 1

    Entities:

    - car 1 starts at ``(0, 0)``, boxed in by the obstacles at ``(0, 1)``
      and ``(1, 0)``; its flag is at ``(2, 2)``, unreachable
    - car 2 starts at ``(2, 0)``, free to move; its own flag sits at
      ``(2, 1)``, one cell to its right and reachable

    Reused by 03-03's ``stranded_cars``/``is_unwinnable`` tests: car 1 is
    the stranded car, car 2 proves the check is per-car, not board-wide.
    """
    board = Board(
        rows=3,
        cols=3,
        obstacles=frozenset({(0, 1), (1, 0)}),
        flags=((2, 2), (2, 1)),
    )
    state = GameState(cars=((0, 0), (2, 0)), parked=frozenset())
    return board, state


@pytest.fixture
def gridlock_board() -> tuple[Board, GameState]:
    """A 2x2 board where both cars are boxed in by each other, with nothing stranded.

    Layout, ``(row, col)`` with row 0 at the top::

        col:      0          1
        row 0:  car 1      car 2
        row 1: flag 2     flag 1

    Neither car has a legal move: car 1's only non-off-grid neighbour
    beyond car 2 (occupied) is car 2's flag (foreign), and symmetrically
    for car 2. This pins branch B of ``is_unwinnable`` (terminal gridlock)
    as independent of branch A (stranded cars): each car's own flag is
    still reachable by the private walk (it can pass through the other,
    still-unparked car to get there), so ``stranded_cars`` is empty here
    even though no move remains.
    """
    board = Board(rows=2, cols=2, obstacles=frozenset(), flags=((1, 1), (1, 0)))
    state = GameState(cars=((0, 0), (0, 1)), parked=frozenset())
    return board, state


@pytest.fixture
def valid_level_dict() -> dict[str, Any]:
    """Return a fresh valid level dictionary.

    Layout (3x3 grid):
    - Car 1 at (0, 0), Flag 1 at (2, 2)
    - Car 2 at (0, 2), Flag 2 at (2, 0)
    - Obstacle at (1, 1)
    """
    return {
        "name": "Fixture Level",
        "rows": 3,
        "cols": 3,
        "obstacles": [[1, 1]],
        "cars": [
            {"number": 1, "at": [0, 0]},
            {"number": 2, "at": [0, 2]},
        ],
        "flags": [
            {"number": 1, "at": [2, 2]},
            {"number": 2, "at": [2, 0]},
        ],
    }


@pytest.fixture
def write_level(tmp_path: Path):
    """Return a helper function that writes a level dictionary as JSON to a temp file."""

    def _writer(data: dict[str, Any], filename: str = "level.json") -> Path:
        p = tmp_path / filename
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    return _writer
