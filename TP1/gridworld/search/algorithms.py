"""Search algorithms and their incremental execution support.

Every function takes a ``Problem`` (plus a heuristic where the algorithm is
informed) and returns a ``SearchResult`` with every metric field populated
(success, cost, path, expanded_nodes, frontier_nodes, elapsed_seconds, max_frontier_nodes). A*
is implemented with an incremental stepper for the UI.
"""

from __future__ import annotations

from collections import deque
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


def bfs(
    problem: Problem, on_expand: Callable[[GameState], None] | None = None
) -> SearchResult:
    """Breadth-first search: uninformed graph search expanding by increasing depth. SRCH-01."""
    start = time.perf_counter()
    root = SearchNode.root(problem.initial)

    if problem.is_goal(root.state):
        return SearchResult(
            algorithm="bfs",
            success=True,
            cost=0,
            path=(),
            expanded_nodes=0,
            frontier_nodes=0,
            elapsed_seconds=time.perf_counter() - start,
            max_frontier_nodes=0,
        )

    frontier: deque[SearchNode] = deque([root])
    frontier_states: set[GameState] = {root.state}
    explored: set[GameState] = set()
    expanded = 0
    max_frontier = 1

    while frontier:
        node = frontier.popleft()
        frontier_states.remove(node.state)
        explored.add(node.state)
        expanded += 1
        if on_expand is not None:
            on_expand(node.state)

        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if child_state in explored or child_state in frontier_states:
                continue

            child_node = SearchNode(
                state=child_state,
                parent=node,
                action=action,
                path_cost=node.path_cost + problem.step_cost(node.state, action, child_state),
                depth=node.depth + 1,
            )

            if problem.is_goal(child_state):
                return SearchResult(
                    algorithm="bfs",
                    success=True,
                    cost=child_node.path_cost,
                    path=child_node.path(),
                    expanded_nodes=expanded,
                    frontier_nodes=len(frontier),
                    elapsed_seconds=time.perf_counter() - start,
                    max_frontier_nodes=max_frontier,
                )

            frontier.append(child_node)
            frontier_states.add(child_state)
            if len(frontier) > max_frontier:
                max_frontier = len(frontier)

    return SearchResult(
        algorithm="bfs",
        success=False,
        cost=None,
        path=(),
        expanded_nodes=expanded,
        frontier_nodes=0,
        elapsed_seconds=time.perf_counter() - start,
        max_frontier_nodes=max_frontier,
    )


def dfs(
    problem: Problem, on_expand: Callable[[GameState], None] | None = None
) -> SearchResult:
    """Depth-first search: uninformed graph search expanding depth-first. SRCH-02."""
    start = time.perf_counter()
    root = SearchNode.root(problem.initial)

    frontier: list[SearchNode] = [root]
    explored: set[GameState] = set()
    expanded = 0
    max_frontier = 1

    while frontier:
        node = frontier.pop()

        if problem.is_goal(node.state):
            return SearchResult(
                algorithm="dfs",
                success=True,
                cost=node.path_cost,
                path=node.path(),
                expanded_nodes=expanded,
                frontier_nodes=len(frontier),
                elapsed_seconds=time.perf_counter() - start,
                max_frontier_nodes=max_frontier,
            )

        if node.state in explored:
            continue

        explored.add(node.state)
        expanded += 1
        if on_expand is not None:
            on_expand(node.state)

        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if child_state not in explored:
                child_node = SearchNode(
                    state=child_state,
                    parent=node,
                    action=action,
                    path_cost=node.path_cost + problem.step_cost(node.state, action, child_state),
                    depth=node.depth + 1,
                )
                frontier.append(child_node)
                if len(frontier) > max_frontier:
                    max_frontier = len(frontier)

    return SearchResult(
        algorithm="dfs",
        success=False,
        cost=None,
        path=(),
        expanded_nodes=expanded,
        frontier_nodes=0,
        elapsed_seconds=time.perf_counter() - start,
        max_frontier_nodes=max_frontier,
    )


def greedy(
    problem: Problem,
    heuristic: Heuristic,
    on_expand: Callable[[GameState], None] | None = None,
) -> SearchResult:
    """Greedy best-first search: informed, orders the frontier by heuristic value alone. SRCH-03."""
    start = time.perf_counter()
    root = SearchNode.root(problem.initial)

    # Entry format: (h(state), tie_breaker_counter, node)
    counter = 0
    heap: list[tuple[float, int, SearchNode]] = [(heuristic(problem, root.state), counter, root)]
    explored: set[GameState] = set()
    expanded = 0
    max_frontier = 1

    while heap:
        _, _, node = heapq.heappop(heap)

        if problem.is_goal(node.state):
            return SearchResult(
                algorithm="greedy",
                success=True,
                cost=node.path_cost,
                path=node.path(),
                expanded_nodes=expanded,
                frontier_nodes=len(heap),
                elapsed_seconds=time.perf_counter() - start,
                max_frontier_nodes=max_frontier,
            )

        if node.state in explored:
            continue

        explored.add(node.state)
        expanded += 1
        if on_expand is not None:
            on_expand(node.state)

        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if child_state not in explored:
                child_node = SearchNode(
                    state=child_state,
                    parent=node,
                    action=action,
                    path_cost=node.path_cost + problem.step_cost(node.state, action, child_state),
                    depth=node.depth + 1,
                )
                counter += 1
                heapq.heappush(heap, (heuristic(problem, child_state), counter, child_node))
                if len(heap) > max_frontier:
                    max_frontier = len(heap)

    return SearchResult(
        algorithm="greedy",
        success=False,
        cost=None,
        path=(),
        expanded_nodes=expanded,
        frontier_nodes=0,
        elapsed_seconds=time.perf_counter() - start,
        max_frontier_nodes=max_frontier,
    )


def astar(
    problem: Problem,
    heuristic: Heuristic,
    on_expand: Callable[[GameState], None] | None = None,
) -> SearchResult:
    """A* search: informed, orders the frontier by path cost plus heuristic value. SRCH-04."""
    stepper = AStarStepper(problem, heuristic)
    while stepper.result is None:
        for expansion in stepper.advance(1_000):
            if on_expand is not None:
                on_expand(expansion.state)
    return stepper.result


def iddfs(
    problem: Problem,
    max_depth: int = 1000,
    on_expand: Callable[[GameState], None] | None = None,
) -> SearchResult:
    """Iterative deepening DFS: uninformed, expands depth-first with increasing limits. SRCH-05."""
    start = time.perf_counter()
    total_expanded = 0
    overall_max_frontier = 1

    for limit in range(max_depth + 1):
        cutoff_occurred = False
        root = SearchNode.root(problem.initial)
        frontier: list[SearchNode] = [root]
        visited_depth: dict[GameState, int] = {root.state: 0}
        iteration_max_frontier = 1

        while frontier:
            node = frontier.pop()

            if problem.is_goal(node.state):
                return SearchResult(
                    algorithm="iddfs",
                    success=True,
                    cost=node.path_cost,
                    path=node.path(),
                    expanded_nodes=total_expanded,
                    frontier_nodes=len(frontier),
                    elapsed_seconds=time.perf_counter() - start,
                    max_frontier_nodes=overall_max_frontier,
                )

            if node.depth >= limit:
                cutoff_occurred = True
                continue

            total_expanded += 1
            if on_expand is not None:
                on_expand(node.state)

            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                child_depth = node.depth + 1

                if child_state in visited_depth and visited_depth[child_state] <= child_depth:
                    continue

                visited_depth[child_state] = child_depth
                child_node = SearchNode(
                    state=child_state,
                    parent=node,
                    action=action,
                    path_cost=node.path_cost + problem.step_cost(node.state, action, child_state),
                    depth=child_depth,
                )
                frontier.append(child_node)
                if len(frontier) > iteration_max_frontier:
                    iteration_max_frontier = len(frontier)
                    if iteration_max_frontier > overall_max_frontier:
                        overall_max_frontier = iteration_max_frontier

        if not cutoff_occurred:
            # Entire state space within reachable depths was explored without finding a goal
            break

    return SearchResult(
        algorithm="iddfs",
        success=False,
        cost=None,
        path=(),
        expanded_nodes=total_expanded,
        frontier_nodes=0,
        elapsed_seconds=time.perf_counter() - start,
        max_frontier_nodes=overall_max_frontier,
    )


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
        self._max_frontier = 1

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
                if len(self._frontier) > self._max_frontier:
                    self._max_frontier = len(self._frontier)

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
            max_frontier_nodes=self._max_frontier,
        )
