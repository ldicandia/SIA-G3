"""A node in a search tree: a state plus the path that reached it.

Generic across every algorithm in ``algorithms.py`` -- BFS, DFS, Greedy,
A*, and IDDFS all build trees out of the same node shape and differ only
in which node their frontier picks next. This module makes no decision
about frontier order, expansion order, or visited-set policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from gridworld.engine.rules import LegalMove
from gridworld.engine.state import GameState


@dataclass(frozen=True, slots=True)
class SearchNode:
    """One node in a search tree.

    ``parent`` is ``None`` only at the root. ``action`` is the move that
    produced ``state`` from ``parent.state`` and is ``None`` only at the
    root. ``path_cost`` is the accumulated cost from the root (``g`` in
    A* terms); ``depth`` is the number of moves from the root.
    """

    state: GameState
    parent: "SearchNode | None"
    action: LegalMove | None
    path_cost: int
    depth: int

    @classmethod
    def root(cls, state: GameState) -> "SearchNode":
        """Return the root node for ``state``: no parent, no action, zero cost."""
        return cls(state=state, parent=None, action=None, path_cost=0, depth=0)

    def path(self) -> tuple[LegalMove, ...]:
        """Return the action sequence from the root to this node, in order."""
        actions: list[LegalMove] = []
        node: SearchNode | None = self
        while node is not None and node.action is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return tuple(actions)
