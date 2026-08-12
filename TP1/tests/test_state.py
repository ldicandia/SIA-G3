"""The immutable, hashable state contract (ENG-07).

These tests exist for the v2 search milestone: BFS, DFS, Greedy, A* and
IDDFS all need game states as visited-set and frontier keys. Every
assertion here is a property a search algorithm silently depends on.

Per decision D-04 this module imports only from the engine.
"""

from __future__ import annotations

import dataclasses

import pytest

from gridworld.engine.rules import apply_move
from gridworld.engine.state import Direction, GameState


def test_equal_states_hash_equal_and_collapse_in_a_set():
    left = GameState(cars=((0, 0), (1, 2)), parked=frozenset({2}))
    right = GameState(cars=((0, 0), (1, 2)), parked=frozenset({2}))

    assert left is not right
    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1


def test_state_usable_as_dict_key():
    """Storing under one instance must read back via an equal instance."""
    key = GameState(cars=((3, 4),), parked=frozenset())
    equal_key = GameState(cars=((3, 4),), parked=frozenset())

    visited = {key: "explored"}

    assert visited[equal_key] == "explored"
    assert equal_key in visited


def test_state_with_empty_parked_set_is_hashable():
    """No code path may assume at least one car has parked."""
    state = GameState(cars=((0, 0), (0, 1)), parked=frozenset())

    assert hash(state) is not None
    assert state.parked == frozenset()
    assert {state: 1}[state] == 1


def test_state_is_frozen_and_rejects_attribute_assignment():
    """A frozen slots dataclass may raise either error; accept both."""
    state = GameState(cars=((0, 0),), parked=frozenset())
    original_cars = state.cars

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        state.cars = ((9, 9),)

    assert state.cars == original_cars


def test_apply_move_does_not_mutate_source_state(tiny_board):
    """The strongest guard in the suite.

    An accepted move must leave the source state byte-identical -- same
    positions, same parked set, same hash -- and return a different
    object. On a rejected move the returned object IS the source, so
    identity there is the expected outcome, not a failure.
    """
    board, state = tiny_board
    cars_before = state.cars
    parked_before = state.parked
    hash_before = hash(state)

    accepted = apply_move(board, state, 2, Direction.RIGHT)
    assert accepted.accepted
    assert accepted.state is not state

    assert state.cars == cars_before
    assert state.parked == parked_before
    assert hash(state) == hash_before

    rejected = apply_move(board, state, 2, Direction.UP)
    assert not rejected.accepted
    assert rejected.state is state
    assert state.cars == cars_before
    assert state.parked == parked_before
    assert hash(state) == hash_before


def test_parked_order_does_not_affect_equality(tiny_board):
    """Protects visited-set dedup for the v2 search milestone.

    Two states reached by parking the same cars in opposite order are one
    position, not two. If they compared unequal, a search would expand
    the same board twice and the visited set would stop working.
    """
    board, start = tiny_board

    first = apply_move(board, start, 1, Direction.DOWN).state
    first = apply_move(board, first, 2, Direction.RIGHT).state

    second = apply_move(board, start, 2, Direction.RIGHT).state
    second = apply_move(board, second, 1, Direction.DOWN).state

    assert first.parked == second.parked
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
