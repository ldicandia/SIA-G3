"""Wiring checks for the search scaffolding (v2 milestone prep).

These tests verify the ``Problem`` adapter and ``SearchNode`` bookkeeping
agree with the engine they wrap, and that the registries are wired. A* is
covered separately; these checks retain contracts for the remaining stubs.
"""

from __future__ import annotations

import pytest

from gridworld.engine.rules import apply_move, is_solved, legal_moves
from gridworld.search.algorithms import ALGORITHMS
from gridworld.search.heuristics import HEURISTICS
from gridworld.search.node import SearchNode
from gridworld.search.problem import Problem


def test_actions_agrees_with_legal_moves(built_in, tiny_board):
    for board, state in (built_in, tiny_board):
        problem = Problem(board=board, initial=state)
        assert problem.actions(state) == legal_moves(board, state)


def test_result_agrees_with_apply_move(built_in):
    board, state = built_in
    problem = Problem(board=board, initial=state)

    for action in problem.actions(state):
        expected = apply_move(board, state, action.car, action.direction).state
        assert problem.result(state, action) == expected


def test_is_goal_agrees_with_is_solved(built_in, solution_sequence):
    board, state = built_in
    problem = Problem(board=board, initial=state)

    assert problem.is_goal(state) == is_solved(board, state)

    for car, direction in solution_sequence:
        result = apply_move(board, state, car, direction)
        assert result.accepted, (car, direction, result.rejection)
        state = result.state
        assert problem.is_goal(state) == is_solved(board, state)


def test_step_cost_is_always_one(built_in):
    board, state = built_in
    problem = Problem(board=board, initial=state)

    for action in problem.actions(state):
        next_state = problem.result(state, action)
        assert problem.step_cost(state, action, next_state) == 1


def test_root_node_has_empty_path(built_in):
    _, state = built_in
    root = SearchNode.root(state)

    assert root.parent is None
    assert root.action is None
    assert root.path_cost == 0
    assert root.depth == 0
    assert root.path() == ()


def test_node_path_reconstructs_action_sequence(built_in):
    board, state = built_in
    problem = Problem(board=board, initial=state)
    root = SearchNode.root(state)

    first_action = problem.actions(state)[0]
    first_state = problem.result(state, first_action)
    child = SearchNode(
        state=first_state,
        parent=root,
        action=first_action,
        path_cost=1,
        depth=1,
    )

    second_action = problem.actions(first_state)[0]
    second_state = problem.result(first_state, second_action)
    grandchild = SearchNode(
        state=second_state,
        parent=child,
        action=second_action,
        path_cost=2,
        depth=2,
    )

    assert grandchild.path() == (first_action, second_action)


@pytest.mark.parametrize("name", ["bfs", "dfs", "greedy", "iddfs"])
def test_algorithm_registry_entries_are_unimplemented_stubs(built_in, name):
    board, state = built_in
    problem = Problem(board=board, initial=state)
    algorithm = ALGORITHMS[name]

    with pytest.raises(NotImplementedError):
        if name in ("greedy", "astar"):
            algorithm(problem, HEURISTICS["heuristic_a"])
        else:
            algorithm(problem)


@pytest.mark.parametrize("name", ["heuristic_b"])
def test_heuristic_registry_entries_are_unimplemented_stubs(built_in, name):
    board, state = built_in
    problem = Problem(board=board, initial=state)
    heuristic = HEURISTICS[name]

    with pytest.raises(NotImplementedError):
        heuristic(problem, state)
