"""Adapter exposing the existing engine as a classic search problem.

Wraps a fixed ``Board`` and initial ``GameState`` behind the four
questions a search algorithm asks: what actions are available from a
state, what state does an action lead to, is a state the goal, and what
does a step cost. Every answer delegates to ``gridworld.engine.rules`` --
this module restates none of the move, parking, or win rules itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from gridworld.engine.board import Board
from gridworld.engine.rules import LegalMove, apply_move, is_solved, legal_moves
from gridworld.engine.state import GameState


@dataclass(frozen=True, slots=True)
class Problem:
    """A Grid World level as a search problem: fixed board, fixed initial state."""

    board: Board
    initial: GameState

    def actions(self, state: GameState) -> tuple[LegalMove, ...]:
        """Every legal move from ``state``. Delegates to ``legal_moves``."""
        return legal_moves(self.board, state)

    def result(self, state: GameState, action: LegalMove) -> GameState:
        """The state reached by applying ``action`` to ``state``.

        ``action`` must be a member of ``self.actions(state)`` -- this is
        not re-validated here, the same contract ``apply_move`` documents
        for an already-legal move.
        """
        return apply_move(self.board, state, action.car, action.direction).state

    def is_goal(self, state: GameState) -> bool:
        """Whether ``state`` is a solved state. Delegates to ``is_solved``."""
        return is_solved(self.board, state)

    def step_cost(self, state: GameState, action: LegalMove, next_state: GameState) -> int:
        """The cost of one accepted move. Uniform: every move costs 1."""
        return 1
