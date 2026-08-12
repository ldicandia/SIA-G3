"""The full move, rejection, parking and win-detection rule matrix (D-04).

Every rejection test asserts the specific ``MoveRejection`` member rather
than merely "not accepted" -- a suite that only checked the boolean would
still pass if every refusal collapsed into a single reason.

Per decision D-04 the renderer is deliberately untested here: this module
imports only from the engine and the level source.
"""

from __future__ import annotations

from gridworld.engine.board import Board
from gridworld.engine.rules import MoveRejection, apply_move, is_solved
from gridworld.engine.state import Direction, GameState


def test_move_shifts_one_car_one_cell_and_returns_new_state(built_in):
    board, state = built_in

    result = apply_move(board, state, 1, Direction.DOWN)

    assert result.accepted
    assert result.rejection is None
    assert result.state is not state
    assert result.state.position_of(1) == (1, 0)
    assert result.state.position_of(2) == state.position_of(2)
    assert result.state.position_of(3) == state.position_of(3)


def test_rejects_target_off_grid(tiny_board):
    """Car 2 sits on the top row, so UP leaves the grid entirely."""
    board, state = tiny_board

    result = apply_move(board, state, 2, Direction.UP)

    assert not result.accepted
    assert result.rejection is MoveRejection.OFF_GRID
    assert result.state is state


def test_rejects_target_obstacle(tiny_board):
    """Car 2 at (0, 1) has the obstacle at (1, 1) directly below it."""
    board, state = tiny_board

    result = apply_move(board, state, 2, Direction.DOWN)

    assert not result.accepted
    assert result.rejection is MoveRejection.OBSTACLE
    assert result.state is state


def test_rejects_target_occupied_by_another_car(tiny_board):
    """Car 2 at (0, 1) has car 1 sitting at (0, 0) to its left."""
    board, state = tiny_board

    result = apply_move(board, state, 2, Direction.LEFT)

    assert not result.accepted
    assert result.rejection is MoveRejection.OCCUPIED
    assert result.state is state


def test_rejects_flag_belonging_to_another_car(tiny_board):
    """Car 2 moved below car 1's flag may not step onto it.

    This is the ENG-03 foreign-flag rule: a flag is passable only by the
    car whose number it carries.
    """
    board, _ = tiny_board
    state = GameState(cars=((0, 0), (2, 0)), parked=frozenset())

    result = apply_move(board, state, 2, Direction.UP)

    assert not result.accepted
    assert result.rejection is MoveRejection.FOREIGN_FLAG
    assert result.state is state


def test_rejects_move_of_parked_car(tiny_board):
    board, state = tiny_board
    parked = apply_move(board, state, 2, Direction.RIGHT)
    assert parked.accepted and parked.state.is_parked(2)

    result = apply_move(board, parked.state, 2, Direction.LEFT)

    assert not result.accepted
    assert result.rejection is MoveRejection.CAR_PARKED
    assert result.state is parked.state


def test_rejects_unknown_car_number(tiny_board):
    board, state = tiny_board

    result = apply_move(board, state, 5, Direction.DOWN)

    assert not result.accepted
    assert result.rejection is MoveRejection.NO_SUCH_CAR
    assert result.state is state


def test_rejection_returns_caller_state_by_identity(tiny_board):
    """Every refusal hands back the caller's own object, not a copy.

    This is what makes a partially applied move unobservable at any call
    site.
    """
    board, state = tiny_board

    for direction in (Direction.UP, Direction.DOWN, Direction.LEFT):
        result = apply_move(board, state, 2, direction)
        assert not result.accepted
        assert result.state is state
        assert result.rejection is not None


def test_car_parks_on_own_flag(tiny_board):
    """Parking is atomic with the move that lands on the flag."""
    board, state = tiny_board

    result = apply_move(board, state, 2, Direction.RIGHT)

    assert result.accepted
    assert result.state.position_of(2) == (0, 2)
    assert result.state.is_parked(2)
    assert 2 in result.state.parked


def test_parked_car_never_moves_again(tiny_board):
    board, state = tiny_board
    parked = apply_move(board, state, 2, Direction.RIGHT).state
    assert parked.is_parked(2)

    for direction in Direction:
        result = apply_move(board, parked, 2, direction)
        assert not result.accepted
        assert result.rejection is MoveRejection.CAR_PARKED
        assert result.state is parked
        assert result.state.position_of(2) == (0, 2)


def test_solved_when_every_car_parked_on_matching_flag(tiny_board):
    board, state = tiny_board

    state = apply_move(board, state, 1, Direction.DOWN).state
    state = apply_move(board, state, 2, Direction.RIGHT).state

    assert state.position_of(1) == board.flag_for(1)
    assert state.position_of(2) == board.flag_for(2)
    assert is_solved(board, state)


def test_not_solved_while_any_car_unparked(tiny_board):
    board, state = tiny_board
    assert not is_solved(board, state)

    state = apply_move(board, state, 2, Direction.RIGHT).state

    assert state.is_parked(2)
    assert not state.is_parked(1)
    assert not is_solved(board, state)


def _drive(board, state, moves):
    for car, direction in moves:
        result = apply_move(board, state, car, direction)
        assert result.accepted, (car, direction, result.rejection)
        state = result.state
    return state


def test_solved_is_independent_of_parking_order(built_in, solution_sequence):
    """Two orderings that park the same cars reach the same solved state.

    Order A parks car 1, then car 3, then car 2. Order B parks car 1,
    then car 2, then car 3. Both must report solved and compare equal.
    """
    board, start = built_in

    order_a = _drive(board, start, solution_sequence)

    order_b_moves = (
        [(1, Direction.DOWN)] * 5
        + [(1, Direction.RIGHT)] * 6
        + [(1, Direction.DOWN)]
        + [(2, Direction.DOWN)] * 5
        + [(2, Direction.LEFT)] * 3
        + [(2, Direction.DOWN)]
        + [(3, Direction.UP)] * 6
        + [(3, Direction.RIGHT)] * 3
    )
    order_b = _drive(board, start, order_b_moves)

    assert is_solved(board, order_a)
    assert is_solved(board, order_b)
    assert order_a == order_b
    assert hash(order_a) == hash(order_b)


def test_solved_single_car_level():
    """is_solved is total over the car set, including a one-car board."""
    board = Board(rows=1, cols=2, obstacles=frozenset(), flags=((0, 1),))
    state = GameState(cars=((0, 0),), parked=frozenset())

    assert not is_solved(board, state)

    result = apply_move(board, state, 1, Direction.RIGHT)

    assert result.accepted
    assert result.state.is_parked(1)
    assert is_solved(board, result.state)
