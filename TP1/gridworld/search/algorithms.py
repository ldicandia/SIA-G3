"""Search algorithms and their incremental execution support.

Every function takes a ``Problem`` (plus a heuristic where the algorithm is
informed) and returns a ``SearchResult``. A* is implemented with an
incremental stepper for the UI; the remaining entries retain their stubs.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import time
from typing import Callable

from gridworld.engine.state import GameState
from gridworld.search.heuristics import Heuristic
from gridworld.search.metrics import SearchResult
from gridworld.search.node import SearchNode
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
    stepper = AStarStepper(problem, heuristic)
    while stepper.result is None:
        stepper.advance(1_000)
    return stepper.result


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


@dataclass(frozen=True, slots=True)
class SearchExpansion:
    """One state removed from A*'s frontier for expansion or goal testing."""

    state: GameState
    expanded_nodes: int
    frontier_nodes: int


class AStarStepper:
    """Incremental A* with the same result contract as :func:`astar`.

    ``advance`` performs a bounded amount of CPU work and returns the states
    inspected during that slice.  The UI uses those states for animation;
    headless callers can simply keep advancing until ``result`` is populated.
    """

    def __init__(self, problem: Problem, heuristic: Heuristic) -> None:
        self.problem = problem
        self.heuristic = heuristic
        self.started_at = time.perf_counter()
        self.expanded_nodes = 0
        self.result: SearchResult | None = None

        root = SearchNode.root(problem.initial)
        self._sequence = itertools.count()
        self._frontier: list[tuple[float, int, int, SearchNode]] = []
        heapq.heappush(
            self._frontier,
            (heuristic(problem, root.state), 0, next(self._sequence), root),
        )
        self._best_cost: dict[GameState, int] = {root.state: 0}

    @property
    def frontier_nodes(self) -> int:
        return len(self._frontier)

    def advance(self, limit: int) -> tuple[SearchExpansion, ...]:
        """Inspect at most ``limit`` live frontier nodes.

        Stale heap entries created by a cheaper replacement do not consume the
        caller's expansion budget and are never exposed as visual events.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if self.result is not None:
            return ()

        events: list[SearchExpansion] = []
        inspected = 0
        while self._frontier and inspected < limit:
            _, path_cost, _, node = heapq.heappop(self._frontier)
            if path_cost != self._best_cost.get(node.state):
                continue

            inspected += 1
            if self.problem.is_goal(node.state):
                self.result = self._make_result(node)
                events.append(
                    SearchExpansion(node.state, self.expanded_nodes, len(self._frontier))
                )
                break

            self.expanded_nodes += 1
            for action in self.problem.actions(node.state):
                child_state = self.problem.result(node.state, action)
                child_cost = path_cost + self.problem.step_cost(
                    node.state, action, child_state
                )
                if child_cost >= self._best_cost.get(child_state, 2**63 - 1):
                    continue
                self._best_cost[child_state] = child_cost
                child = SearchNode(
                    state=child_state,
                    parent=node,
                    action=action,
                    path_cost=child_cost,
                    depth=node.depth + 1,
                )
                priority = child_cost + self.heuristic(self.problem, child_state)
                heapq.heappush(
                    self._frontier,
                    (priority, child_cost, next(self._sequence), child),
                )

            events.append(
                SearchExpansion(node.state, self.expanded_nodes, len(self._frontier))
            )

        if not self._frontier and self.result is None:
            self.result = self._make_result(None)
        return tuple(events)

    def _make_result(self, goal: SearchNode | None) -> SearchResult:
        return SearchResult(
            algorithm="astar",
            success=goal is not None,
            cost=goal.path_cost if goal is not None else None,
            path=goal.path() if goal is not None else (),
            expanded_nodes=self.expanded_nodes,
            frontier_nodes=len(self._frontier),
            elapsed_seconds=time.perf_counter() - self.started_at,
        )
