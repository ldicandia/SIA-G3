"""Heuristic function stubs for the informed search algorithms.

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
    """First admissible heuristic. HEUR-01."""
    raise NotImplementedError("HEUR-01")


def heuristic_b(problem: Problem, state: GameState) -> float:
    """Second admissible heuristic. HEUR-01."""
    raise NotImplementedError("HEUR-01")


HEURISTICS: dict[str, Heuristic] = {
    "heuristic_a": heuristic_a,
    "heuristic_b": heuristic_b,
}
