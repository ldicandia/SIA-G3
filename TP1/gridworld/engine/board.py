"""Static board layout: grid dimensions, obstacles, and numbered flags.

The board never changes during play. Car positions and parked status live
in ``gridworld.engine.state.GameState`` instead, so a search node only
carries what actually varies from move to move.
"""

from __future__ import annotations

from dataclasses import dataclass

Position = tuple[int, int]
"""A grid cell as ``(row, col)``, with row 0 at the top."""


@dataclass(frozen=True, slots=True)
class Board:
    """The static layout of a Grid World level.

    ``flags[i]`` is the destination of car number ``i + 1``: car numbers
    are 1-based everywhere a caller touches them, and the 0-based tuple
    translation lives only inside this class's accessors.
    """

    rows: int
    cols: int
    obstacles: frozenset[Position]
    flags: tuple[Position, ...]

    def car_numbers(self) -> range:
        """Return the 1-based car numbers this board defines."""
        return range(1, len(self.flags) + 1)

    def flag_for(self, car: int) -> Position:
        """Return the destination flag position for ``car``."""
        return self.flags[car - 1]

    def in_bounds(self, pos: Position) -> bool:
        """Return whether ``pos`` lies within the grid boundary."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_obstacle(self, pos: Position) -> bool:
        """Return whether ``pos`` holds a black obstacle cell."""
        return pos in self.obstacles

    def flag_owner(self, pos: Position) -> int | None:
        """Return the car number owning the flag at ``pos``, or ``None``."""
        for index, flag_pos in enumerate(self.flags):
            if flag_pos == pos:
                return index + 1
        return None
