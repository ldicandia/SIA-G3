"""The full move, rejection, parking and win-detection rule matrix (D-04).

Every rejection test asserts the specific ``MoveRejection`` member rather
than merely "not accepted" -- a suite that only checked the boolean would
still pass if every refusal collapsed into a single reason.

Per decision D-04 the renderer is deliberately untested here: this module
imports only from the engine and the level source.
"""

from __future__ import annotations

from gridworld.engine.board import Board
from gridworld.engine.rules import (
    LegalMove,
    MoveRejection,
    apply_move,
    is_solved,
    legal_moves,
    legal_moves_for,
)
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


def test_legal_moves_ordered_by_ascending_car_then_direction_declaration_order(tiny_board):
    """Each car on tiny_board has exactly one legal move, car 1 before car 2.

    ``legal_moves`` must delegate to ``apply_move``, so the two entries are
    exactly the two accepted moves from the tiny_board layout: car 1 down
    onto its own flag, car 2 right onto its own flag.
    """
    board, state = tiny_board

    result = legal_moves(board, state)

    assert result == (
        LegalMove(car=1, direction=Direction.DOWN, target=(1, 0)),
        LegalMove(car=2, direction=Direction.RIGHT, target=(0, 2)),
    )


def test_legal_moves_within_car_follows_direction_declaration_order():
    """A car with all four directions legal lists them UP, DOWN, LEFT, RIGHT.

    Car 1 sits in the middle of an open 3x3 board with its own flag one
    cell above it and open cells in every other direction, so all four
    directions are accepted and the returned order pins the declaration
    order guarantee directly.
    """
    board = Board(rows=3, cols=3, obstacles=frozenset(), flags=((0, 1), (2, 2)))
    state = GameState(cars=((1, 1), (2, 0)), parked=frozenset())

    result = legal_moves_for(board, state, 1)

    assert result == (
        LegalMove(car=1, direction=Direction.UP, target=(0, 1)),
        LegalMove(car=1, direction=Direction.DOWN, target=(2, 1)),
        LegalMove(car=1, direction=Direction.LEFT, target=(1, 0)),
        LegalMove(car=1, direction=Direction.RIGHT, target=(1, 2)),
    )


def test_legal_moves_empty_when_every_car_parked():
    """An all-parked board enumerates to an empty tuple, never ``None``."""
    board = Board(rows=2, cols=2, obstacles=frozenset(), flags=((0, 0), (1, 1)))
    state = GameState(cars=((0, 0), (1, 1)), parked=frozenset({1, 2}))

    result = legal_moves(board, state)

    assert result == ()
    assert isinstance(result, tuple)


def test_legal_moves_empty_when_every_car_boxed_in():
    """Corridor deadlock fixture, reused by 03-03's unwinnable detection.

    A 1x3 corridor: car 1 at (0,0) heading for flag 1 at (0,2); car 2 at
    (0,1) heading for flag 2 at (0,0). Car 1 is blocked by car 2 to its
    right and the grid edge everywhere else. Car 2 is blocked by car 1
    (occupied) to its left and by car 1's foreign flag to its right.
    """
    board = Board(rows=1, cols=3, obstacles=frozenset(), flags=((0, 2), (0, 0)))
    state = GameState(cars=((0, 0), (0, 1)), parked=frozenset())

    result = legal_moves(board, state)

    assert result == ()


def test_legal_moves_for_empty_for_parked_car_on_single_car_board():
    board = Board(rows=1, cols=1, obstacles=frozenset(), flags=((0, 0),))
    state = GameState(cars=((0, 0),), parked=frozenset({1}))

    assert legal_moves_for(board, state, 1) == ()


def test_legal_moves_for_empty_for_unknown_car_number(tiny_board):
    board, state = tiny_board

    assert legal_moves_for(board, state, 99) == ()


def test_legal_moves_adjacency_never_targets_the_other_cars_cell():
    """Two orthogonally adjacent cars never enumerate a move onto each other.

    Car 1 and car 2 sit side by side with their flags placed away from the
    shared boundary, so every accepted move is one this test can check by
    hand: the total entry count is exactly the sum of each car's own count,
    and no entry's target is the other car's occupied cell.
    """
    board = Board(rows=3, cols=3, obstacles=frozenset(), flags=((0, 0), (0, 2)))
    state = GameState(cars=((1, 0), (1, 1)), parked=frozenset())

    car_1_moves = legal_moves_for(board, state, 1)
    car_2_moves = legal_moves_for(board, state, 2)
    all_moves = legal_moves(board, state)

    assert car_1_moves == (
        LegalMove(car=1, direction=Direction.UP, target=(0, 0)),
        LegalMove(car=1, direction=Direction.DOWN, target=(2, 0)),
    )
    assert car_2_moves == (
        LegalMove(car=2, direction=Direction.UP, target=(0, 1)),
        LegalMove(car=2, direction=Direction.DOWN, target=(2, 1)),
        LegalMove(car=2, direction=Direction.RIGHT, target=(1, 2)),
    )
    assert all_moves == car_1_moves + car_2_moves
    assert len(all_moves) == len(car_1_moves) + len(car_2_moves)
    assert all(move.target != (1, 1) for move in car_1_moves)
    assert all(move.target != (1, 0) for move in car_2_moves)


def test_legal_moves_adjacent_to_own_flag_produces_exactly_one_entry():
    """A car standing next to its own flag has exactly one legal move: onto it."""
    board = Board(rows=2, cols=1, obstacles=frozenset(), flags=((0, 0),))
    state = GameState(cars=((1, 0),), parked=frozenset())

    result = legal_moves_for(board, state, 1)

    assert result == (LegalMove(car=1, direction=Direction.UP, target=(0, 0)),)
    assert result[0].target == board.flag_for(1)


def _assert_legal_moves_agree_with_apply_move(board, state):
    """Anti-drift check: legal_moves and apply_move must never disagree.

    For every (car, direction) pair, the pair appears in legal_moves if
    and only if apply_move accepts it, and every enumerated entry's target
    matches where apply_move actually lands the car.
    """
    enumerated = {(move.car, move.direction): move for move in legal_moves(board, state)}

    for car in board.car_numbers():
        for direction in Direction:
            result = apply_move(board, state, car, direction)
            pair = (car, direction)
            if result.accepted:
                assert pair in enumerated, f"{pair} accepted by apply_move but missing from legal_moves"
                assert result.state.position_of(car) == enumerated[pair].target
            else:
                assert pair not in enumerated, f"{pair} rejected by apply_move but present in legal_moves"


def test_legal_moves_agrees_with_apply_move_on_built_in_board(built_in):
    board, state = built_in
    _assert_legal_moves_agree_with_apply_move(board, state)


def test_legal_moves_agrees_with_apply_move_on_tiny_board(tiny_board):
    board, state = tiny_board
    _assert_legal_moves_agree_with_apply_move(board, state)


def test_legal_moves_agrees_with_apply_move_including_foreign_flag(tiny_board):
    """The agreement check must also cover a FOREIGN_FLAG rejection.

    Neither fixture's own start state reaches a foreign flag, so this test
    repositions car 2 next to car 1's flag directly -- the same layout
    ``test_rejects_flag_belonging_to_another_car`` uses. Without this case
    a delegation that quietly re-implements (and drifts from) the
    foreign-flag rule would pass the other two agreement tests silently.
    """
    board, _ = tiny_board
    state = GameState(cars=((0, 0), (2, 0)), parked=frozenset())
    _assert_legal_moves_agree_with_apply_move(board, state)


def test_solved_single_car_level():
    """is_solved is total over the car set, including a one-car board."""
    board = Board(rows=1, cols=2, obstacles=frozenset(), flags=((0, 1),))
    state = GameState(cars=((0, 0),), parked=frozenset())

    assert not is_solved(board, state)

    result = apply_move(board, state, 1, Direction.RIGHT)

    assert result.accepted
    assert result.state.is_parked(1)
    assert is_solved(board, result.state)
