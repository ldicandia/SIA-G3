"""Level definitions.

``built_in_level`` is the seam Phase 2's JSON loader replaces without
touching the engine or the renderer. Board construction never happens in
the ui layer.
"""

from __future__ import annotations

from gridworld.engine.board import Board
from gridworld.engine.state import GameState


def built_in_level() -> tuple[Board, GameState]:
    """Return the fixed Phase 1 board: a 7x7 grid with three cars.

    Small, has multiple routes per car, and contains no parking-order
    trap -- Phase 1 ships no undo and no unwinnable warning, so ``R`` is
    the player's only escape.
    """
    board = Board(
        rows=7,
        cols=7,
        obstacles=frozenset({(2, 2), (2, 4), (4, 2), (4, 4), (3, 1), (3, 5)}),
        flags=((6, 6), (6, 3), (0, 3)),
    )
    state = GameState(
        cars=((0, 0), (0, 6), (6, 0)),
        parked=frozenset(),
    )
    return board, state
