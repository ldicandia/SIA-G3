"""Proves the search scaffolding actually works end to end.

``random_search`` is a throwaway demo algorithm (see ``demo.py``), not one
of the graded SRCH-01..05 algorithms. This test exists to exercise
``Problem``, ``SearchNode``, and ``SearchResult`` together against a real
board, not to validate any search strategy.
"""

from __future__ import annotations

from gridworld.engine.rules import apply_move, is_solved
from gridworld.search.demo import random_search
from gridworld.search.problem import Problem


def test_random_search_finds_and_replays_a_solution_on_tiny_board(tiny_board):
    """Uses ``tiny_board``, not ``built_in``: with no goal-directed guidance,
    unordered random expansion needs to cover most of the reachable state
    space before it happens onto a solved state, and ``built_in``'s 7x7
    three-car board is large enough that a bounded budget isn't reliably
    enough. ``tiny_board``'s 3x3 two-car space is small enough to always
    finish fast -- exactly what a plumbing-only demo needs.
    """
    board, state = tiny_board
    problem = Problem(board=board, initial=state)

    result = random_search(problem, seed="search-demo")

    assert result.success
    assert result.algorithm == "random"
    assert result.cost == len(result.path)
    assert result.expanded_nodes > 0

    replayed = state
    for action in result.path:
        outcome = apply_move(board, replayed, action.car, action.direction)
        assert outcome.accepted, (action.car, action.direction, outcome.rejection)
        replayed = outcome.state

    assert is_solved(board, replayed)
