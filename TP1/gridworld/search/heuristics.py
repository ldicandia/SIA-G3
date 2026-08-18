"""Heuristic functions for the informed search algorithms.

A heuristic estimates the remaining cost from a state to the goal. Per
the enunciado, at least two must be admissible -- never overestimate the
true remaining cost -- and non-admissible heuristics are optional, for
comparison only. Neither stub's estimate logic is implemented; that
design is left to be written directly against this signature.
"""

from __future__ import annotations

from typing import Callable

from gridworld.engine.state import GameState
from gridworld.search.problem import Problem

Heuristic = Callable[[Problem, GameState], float]


def heuristic_a(problem: Problem, state: GameState) -> float:
    """Sum each unparked car's Manhattan distance to its own flag.

    Every accepted move changes exactly one car by one orthogonal cell, so
    this ignores obstacles and traffic and can never overestimate the moves
    still required.  It is therefore admissible for A*.
    """
    distance = 0
    for car in problem.board.car_numbers():
        if state.is_parked(car):
            continue
        row, col = state.position_of(car)
        flag_row, flag_col = problem.board.flag_for(car)
        distance += abs(row - flag_row) + abs(col - flag_col)
    return float(distance)


def heuristic_b(problem: Problem, state: GameState) -> float:
    """Second admissible heuristic. HEUR-01."""
    raise NotImplementedError("HEUR-01")


HEURISTICS: dict[str, Heuristic] = {
    "heuristic_a": heuristic_a,
    "heuristic_b": heuristic_b,
}
