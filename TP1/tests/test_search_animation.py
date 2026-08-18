"""The in-game search animation reaches and preserves the optimal route."""

from gridworld.engine.rules import is_solved
from gridworld.levelfile import load_level
from gridworld.ui.app import (
    SearchPhase,
    Session,
    _advance_search,
    _begin_search,
    _start_level,
)


def test_search_animation_explores_then_replays_optimal_path():
    level = load_level("levels/01-warmup.json")
    session = Session()
    _start_level(session, level)
    _begin_search(session)

    for tick in range(1, 2_000):
        _advance_search(session, tick * 1_000)
        if session.search is not None and session.search.phase is SearchPhase.COMPLETE:
            break

    assert session.search is not None
    assert session.search.phase is SearchPhase.COMPLETE
    assert session.search.explored_cells
    assert session.search.solution_paths
    assert session.history is not None
    assert session.history.depth == 16
    assert is_solved(level.board, session.history.current)
