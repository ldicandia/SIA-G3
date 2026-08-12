"""Dynamic game state: car positions and parked status.

``GameState`` is the thing that must be immutable and hashable, and the
thing a future search node holds. It carries no move count, no selection,
and no other per-session or presentation field -- see this phase's plan
prohibitions on ``ENG-07``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gridworld.engine.board import Position


class Direction(Enum):
    """One of the four orthogonal directions a car can move."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

    @property
    def delta(self) -> tuple[int, int]:
        """Return the ``(drow, dcol)`` offset for this direction.

        Row 0 is the top of the grid, so ``UP`` decreases the row.
        """
        return {
            Direction.UP: (-1, 0),
            Direction.DOWN: (1, 0),
            Direction.LEFT: (0, -1),
            Direction.RIGHT: (0, 1),
        }[self]


@dataclass(frozen=True, slots=True)
class GameState:
    """The dynamic state of a Grid World level.

    ``cars[i]`` is the position of car number ``i + 1``. ``parked`` is a
    frozenset (not a tuple) of 1-based car numbers specifically so that
    two states reached by parking the same cars in a different order are
    equal and hash equal.
    """

    cars: tuple[Position, ...]
    parked: frozenset[int]

    def position_of(self, car: int) -> Position:
        """Return the current position of ``car``."""
        return self.cars[car - 1]

    def car_at(self, pos: Position) -> int | None:
        """Return the car number occupying ``pos``, or ``None``."""
        for index, car_pos in enumerate(self.cars):
            if car_pos == pos:
                return index + 1
        return None

    def is_parked(self, car: int) -> bool:
        """Return whether ``car`` has already parked on its flag."""
        return car in self.parked
