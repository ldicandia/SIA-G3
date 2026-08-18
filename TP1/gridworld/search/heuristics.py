"""Heuristic functions for the informed search algorithms (Greedy, A*).

A heuristic estimates the remaining cost from a state to the goal.
Per the enunciado, at least two admissible heuristics are provided.
"""

from __future__ import annotations

from typing import Callable

from gridworld.engine.state import GameState
from gridworld.search.problem import Problem

Heuristic = Callable[[Problem, GameState], float]


def manhattan_distance_sum(problem: Problem, state: GameState) -> float:
    """Admissible heuristic: Sum of Manhattan distances from each unparked car to its flag.

    Since only one car moves 1 cell per step, the total remaining steps must be
    at least the sum of the Manhattan distances of all unparked cars to their destinations.
    """
    total = 0.0
    for car in problem.board.car_numbers():
        if state.is_parked(car):
            continue
        car_pos = state.position_of(car)
        flag_pos = problem.board.flag_for(car)
        total += abs(car_pos[0] - flag_pos[0]) + abs(car_pos[1] - flag_pos[1])
    return total


def max_manhattan_distance(problem: Problem, state: GameState) -> float:
    """Admissible heuristic: Maximum Manhattan distance among all unparked cars to their flags."""
    max_d = 0.0
    for car in problem.board.car_numbers():
        if state.is_parked(car):
            continue
        car_pos = state.position_of(car)
        flag_pos = problem.board.flag_for(car)
        d = abs(car_pos[0] - flag_pos[0]) + abs(car_pos[1] - flag_pos[1])
        if d > max_d:
            max_d = float(d)
    return max_d


heuristic_a = manhattan_distance_sum
heuristic_b = max_manhattan_distance

HEURISTICS: dict[str, Heuristic] = {
    "heuristic_a": heuristic_a,
    "heuristic_b": heuristic_b,
    "manhattan": manhattan_distance_sum,
    "max_manhattan": max_manhattan_distance,
}
