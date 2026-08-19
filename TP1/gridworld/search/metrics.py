"""The end-of-run report every algorithm in ``algorithms.py`` must produce.

Field names and meaning are pinned directly to the enunciado's reporting
requirement (``docs/Enunciado TP1(1)-2.md``): result (success/failure),
solution cost, expanded node count, frontier node count, solution path,
and processing time. This module holds only the data shape -- no
formatting, printing, or cross-run comparison logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from gridworld.engine.rules import LegalMove


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The outcome of one run of one search algorithm over one ``Problem``.

    ``success`` is ``False`` when the frontier exhausted without reaching
    a goal state; ``cost`` and ``path`` carry no meaning in that case.
    """

    algorithm: str
    success: bool
    cost: int | None
    path: tuple[LegalMove, ...]
    expanded_nodes: int
    frontier_nodes: int
    elapsed_seconds: float
    max_frontier_nodes: int = 0
