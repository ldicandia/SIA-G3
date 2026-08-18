"""Optimal A* search and its incremental visualization contract."""

from gridworld.engine.rules import apply_move, is_solved
from gridworld.search.algorithms import AStarStepper, astar
from gridworld.search.heuristics import heuristic_a
from gridworld.search.problem import Problem


def test_manhattan_heuristic_is_zero_at_goal(built_in, solution_sequence):
    board, state = built_in
    problem = Problem(board, state)

    for car, direction in solution_sequence:
        state = apply_move(board, state, car, direction).state

    assert is_solved(board, state)
    assert heuristic_a(problem, state) == 0


def test_astar_finds_the_known_optimal_built_in_solution(built_in):
    board, initial = built_in
    result = astar(Problem(board, initial), heuristic_a)

    assert result.success
    assert result.algorithm == "astar"
    assert result.cost == 30
    assert len(result.path) == 30

    state = initial
    for action in result.path:
        outcome = apply_move(board, state, action.car, action.direction)
        assert outcome.accepted
        state = outcome.state
    assert is_solved(board, state)


def test_astar_stepper_exposes_expanded_states_before_result(tiny_board):
    board, initial = tiny_board
    stepper = AStarStepper(Problem(board, initial), heuristic_a)
    events = []

    while stepper.result is None:
        events.extend(stepper.advance(1))

    assert events
    assert events[0].state == initial
    assert stepper.result is not None
    assert stepper.result.success
    assert stepper.result.cost == 2
    assert events[-1].state.parked == frozenset({1, 2})
