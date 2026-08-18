"""Tests for search algorithms in gridworld.search.algorithms."""

from __future__ import annotations

import pytest

from gridworld.engine.rules import apply_move, is_solved
from gridworld.search.algorithms import ALGORITHMS, astar, bfs, dfs, greedy, iddfs
from gridworld.search.heuristics import Heuristic
from gridworld.search.problem import Problem


def zero_heuristic(problem: Problem, state) -> float:
    """Trivial admissible heuristic: always returns 0."""
    return 0.0


def test_bfs_finds_optimal_solution(tiny_board):
    board, state = tiny_board
    problem = Problem(board=board, initial=state)

    result = bfs(problem)

    assert result.success
    assert result.algorithm == "bfs"
    assert result.cost == len(result.path)
    assert result.cost is not None
    assert result.expanded_nodes > 0
    assert result.elapsed_seconds >= 0

    # Replay solution
    curr = state
    for action in result.path:
        outcome = apply_move(board, curr, action.car, action.direction)
        assert outcome.accepted
        curr = outcome.state
    assert is_solved(board, curr)


def test_dfs_finds_valid_solution(tiny_board):
    board, state = tiny_board
    problem = Problem(board=board, initial=state)

    result = dfs(problem)

    assert result.success
    assert result.algorithm == "dfs"
    assert result.cost == len(result.path)
    assert result.cost is not None

    curr = state
    for action in result.path:
        outcome = apply_move(board, curr, action.car, action.direction)
        assert outcome.accepted
        curr = outcome.state
    assert is_solved(board, curr)


def test_greedy_finds_valid_solution(tiny_board):
    board, state = tiny_board
    problem = Problem(board=board, initial=state)

    result = greedy(problem, zero_heuristic)

    assert result.success
    assert result.algorithm == "greedy"
    assert result.cost == len(result.path)

    curr = state
    for action in result.path:
        outcome = apply_move(board, curr, action.car, action.direction)
        assert outcome.accepted
        curr = outcome.state
    assert is_solved(board, curr)


def test_astar_finds_optimal_solution(tiny_board):
    board, state = tiny_board
    problem = Problem(board=board, initial=state)

    bfs_res = bfs(problem)
    astar_res = astar(problem, zero_heuristic)

    assert astar_res.success
    assert astar_res.algorithm == "astar"
    assert astar_res.cost == bfs_res.cost  # With zero heuristic, A* finds optimal cost same as BFS

    curr = state
    for action in astar_res.path:
        outcome = apply_move(board, curr, action.car, action.direction)
        assert outcome.accepted
        curr = outcome.state
    assert is_solved(board, curr)


def test_iddfs_finds_optimal_solution(tiny_board):
    board, state = tiny_board
    problem = Problem(board=board, initial=state)

    bfs_res = bfs(problem)
    iddfs_res = iddfs(problem)

    assert iddfs_res.success
    assert iddfs_res.algorithm == "iddfs"
    assert iddfs_res.cost == bfs_res.cost

    curr = state
    for action in iddfs_res.path:
        outcome = apply_move(board, curr, action.car, action.direction)
        assert outcome.accepted
        curr = outcome.state
    assert is_solved(board, curr)


def test_algorithms_registry():
    assert "bfs" in ALGORITHMS
    assert "dfs" in ALGORITHMS
    assert "greedy" in ALGORITHMS
    assert "astar" in ALGORITHMS
    assert "iddfs" in ALGORITHMS
