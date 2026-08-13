"""The move-history contract (UI-07, UI-08): undo, reset, and the promote invariant.

Per decision D-04 this module imports only from the history and engine
layers -- the renderer package is deliberately out of reach here. It
constructs no surface and opens no display.
"""

from __future__ import annotations

import pytest

from gridworld.engine.rules import apply_move
from gridworld.engine.state import Direction, GameState
from gridworld.history import MoveHistory


def test_start_has_depth_zero_and_current_is_the_seeded_state():
    state = GameState(cars=((0, 0), (0, 1)), parked=frozenset())

    history = MoveHistory.start(state)

    assert history.depth == 0
    assert history.current is state
    assert history.initial is state


def test_push_returns_new_instance_and_leaves_receiver_unchanged(built_in):
    board, state = built_in
    history = MoveHistory.start(state)

    states_before = history.states
    depth_before = history.depth
    current_before = history.current

    moved = apply_move(board, state, 1, Direction.DOWN)
    assert moved.accepted
    pushed = history.push(moved.state)

    assert pushed is not history
    assert pushed.depth == depth_before + 1
    assert pushed.current is moved.state

    # The receiver is untouched by push.
    assert history.states == states_before
    assert history.depth == depth_before
    assert history.current is current_before


def test_undo_restores_the_exact_prior_state(built_in):
    board, state = built_in
    history = MoveHistory.start(state)

    moved = apply_move(board, state, 1, Direction.DOWN)
    assert moved.accepted
    history = history.push(moved.state)

    restored = history.undo()

    assert restored.current == state
    assert hash(restored.current) == hash(state)
    assert restored.current is history.states[0]


def test_undo_restores_parked_status(tiny_board):
    """Undoing the move that parked a car un-parks it (ROADMAP success criterion 2)."""
    board, state = tiny_board
    history = MoveHistory.start(state)

    parked_before = state.parked

    moved = apply_move(board, state, 1, Direction.DOWN)
    assert moved.accepted
    assert moved.state.is_parked(1)
    history = history.push(moved.state)

    restored = history.undo()

    assert not restored.current.is_parked(1)
    assert restored.current.parked == parked_before
    assert restored.current.parked != moved.state.parked


def test_repeated_undo_walks_back_to_initial_then_returns_self_by_identity(built_in):
    board, state = built_in
    history = MoveHistory.start(state)

    for car, direction in [(1, Direction.DOWN), (1, Direction.DOWN), (1, Direction.RIGHT)]:
        result = apply_move(board, history.current, car, direction)
        assert result.accepted
        history = history.push(result.state)

    assert history.depth == 3

    history = history.undo()
    history = history.undo()
    history = history.undo()
    assert history.depth == 0
    assert history.current is state

    no_op = history.undo()
    assert no_op is history


def test_reset_from_a_deep_history_lands_on_initial_at_depth_zero(built_in):
    board, state = built_in
    history = MoveHistory.start(state)

    for car, direction in [(1, Direction.DOWN), (1, Direction.DOWN)]:
        result = apply_move(board, history.current, car, direction)
        assert result.accepted
        history = history.push(result.state)

    reset_history = history.reset()

    assert reset_history.depth == 0
    assert reset_history.current is history.initial
    assert reset_history.current is state


def test_reset_at_depth_zero_returns_self_by_identity():
    state = GameState(cars=((0, 0),), parked=frozenset())
    history = MoveHistory.start(state)

    assert history.reset() is history


def test_promote_invariant_holds_after_every_push_undo_reset(built_in):
    """Depth is structurally the history's own length, not a parallel counter.

    This test goes red the instant a future phase reintroduces an
    independent move counter that could disagree with the history.
    """
    board, state = built_in
    history = MoveHistory.start(state)

    def check(h: MoveHistory) -> None:
        assert h.depth == len(h.states) - 1
        assert h.current is h.states[-1]

    check(history)

    script = [(1, Direction.DOWN), (1, Direction.DOWN), (1, Direction.RIGHT)]
    for car, direction in script:
        result = apply_move(board, history.current, car, direction)
        assert result.accepted
        history = history.push(result.state)
        check(history)

    history = history.undo()
    check(history)

    history = history.undo()
    check(history)

    result = apply_move(board, history.current, 1, Direction.DOWN)
    assert result.accepted
    history = history.push(result.state)
    check(history)

    history = history.reset()
    check(history)

    # A no-op reset and a no-op undo also preserve the invariant.
    history = history.reset()
    check(history)
    history = history.undo()
    check(history)


def test_move_history_is_hashable_and_equal_histories_hash_equal():
    left = MoveHistory(states=(GameState(cars=((0, 0),), parked=frozenset()),))
    right = MoveHistory(states=(GameState(cars=((0, 0),), parked=frozenset()),))

    assert left is not right
    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1


def test_empty_state_tuple_raises_value_error():
    with pytest.raises(ValueError):
        MoveHistory(states=())
