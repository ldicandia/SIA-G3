"""The session's move history: a sequence of ``GameState``, not a single one.

This is session narrative -- a record of where this particular human has
been -- not game rules, and it deliberately lives outside
``gridworld/engine/``: the engine's job is "given a state, what may happen
next", and the deferred v2 search milestone will build its own frontier
rather than reuse a player's undo stack. It is nonetheless pygame-free, so
the same clean-venv guarantee (``check_headless.py``, ENG-08) covers it.

``MoveHistory`` is not a search frontier: it never branches, it holds no
visited set, and it is not expanded. It is a plain linear record of the
one path a human actually walked, capped only by however long a single
play session runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from gridworld.engine.state import GameState


@dataclass(frozen=True, slots=True)
class MoveHistory:
    """A level's move history: ``states[0]`` is the opening position,
    ``states[-1]`` is the current position.

    Every ``GameState`` in ``states`` was produced by ``apply_move`` --
    this module constructs none itself. All operations return a new frozen
    instance; the receiver is never modified in place.
    """

    states: tuple[GameState, ...]

    def __post_init__(self) -> None:
        if len(self.states) == 0:
            raise ValueError("MoveHistory requires at least one state")

    @classmethod
    def start(cls, state: GameState) -> "MoveHistory":
        """Return a history of exactly ``state``, the level's opening position."""
        return cls(states=(state,))

    @property
    def current(self) -> GameState:
        """The current position: ``states[-1]``."""
        return self.states[-1]

    @property
    def initial(self) -> GameState:
        """The level's opening position: ``states[0]``."""
        return self.states[0]

    @property
    def depth(self) -> int:
        """The accepted-move count: ``len(states) - 1``.

        This *is* the move counter -- not a parallel copy of it -- so the
        displayed count cannot drift from the history's own length.
        """
        return len(self.states) - 1

    def push(self, state: GameState) -> "MoveHistory":
        """Return a new history with ``state`` appended. Never modifies ``self``."""
        return MoveHistory(states=self.states + (state,))

    def undo(self) -> "MoveHistory":
        """Return a new history with the last state dropped.

        When ``depth`` is already 0 there is nothing to undo, so this
        returns ``self`` **by identity** -- never ``None``, never a copy --
        mirroring the rejected-move contract in
        ``gridworld.engine.rules.MoveResult``. The caller detects the no-op
        with an identity check (``result is history``), exactly as the
        move path does for a rejected ``apply_move``.
        """
        if self.depth == 0:
            return self
        return MoveHistory(states=self.states[:-1])

    def reset(self) -> "MoveHistory":
        """Return a history containing only ``initial``.

        When ``depth`` is already 0 there is nothing to reset, so this
        returns ``self`` **by identity**, the same no-op contract as
        ``undo``.
        """
        if self.depth == 0:
            return self
        return MoveHistory(states=(self.initial,))
