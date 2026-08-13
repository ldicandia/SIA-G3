"""Search algorithm stubs.

Every function takes a ``Problem`` (plus a heuristic where the algorithm
is informed) and must return a ``SearchResult`` with every field
populated. Each docstring names only what the enunciado already
specifies about the algorithm -- not an implementation strategy. None
are implemented yet; that is deliberately left to be written directly
against these signatures.
"""

from __future__ import annotations

from typing import Callable

from gridworld.search.heuristics import Heuristic
from gridworld.search.metrics import SearchResult
from gridworld.search.problem import Problem


def bfs(problem: Problem) -> SearchResult:
    """Breadth-first search: uninformed, expands by increasing depth. SRCH-01."""
    raise NotImplementedError("SRCH-01")


def dfs(problem: Problem) -> SearchResult:
    """Depth-first search: uninformed, expands depth-first. SRCH-02."""
    raise NotImplementedError("SRCH-02")


def greedy(problem: Problem, heuristic: Heuristic) -> SearchResult:
    """Greedy best-first search: informed, orders the frontier by heuristic value alone. SRCH-03."""
    raise NotImplementedError("SRCH-03")


def astar(problem: Problem, heuristic: Heuristic) -> SearchResult:
    """A* search: informed, orders the frontier by path cost plus heuristic value. SRCH-04."""
    raise NotImplementedError("SRCH-04")


def iddfs(problem: Problem) -> SearchResult:
    """Iterative deepening DFS: uninformed, optional per the enunciado. SRCH-05."""
    raise NotImplementedError("SRCH-05")


ALGORITHMS: dict[str, Callable[..., SearchResult]] = {
    "bfs": bfs,
    "dfs": dfs,
    "greedy": greedy,
    "astar": astar,
    "iddfs": iddfs,
}
