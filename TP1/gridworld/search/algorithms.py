"""Search algorithm implementations.

Every function takes a ``Problem`` (plus a heuristic where the algorithm
is informed) and returns a ``SearchResult`` with every metric field
populated (success, cost, path, expanded_nodes, frontier_nodes, elapsed_seconds).
"""

from __future__ import annotations

from collections import deque
import heapq
import time
from typing import Callable

from gridworld.engine.state import GameState
from gridworld.search.heuristics import Heuristic
from gridworld.search.metrics import SearchResult
from gridworld.search.node import SearchNode
from gridworld.search.problem import Problem


def bfs(problem: Problem) -> SearchResult:
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
        )

    frontier: deque[SearchNode] = deque([root])
    frontier_states: set[GameState] = {root.state}
    explored: set[GameState] = set()
    expanded = 0

    while frontier:
        node = frontier.popleft()
        frontier_states.remove(node.state)
        explored.add(node.state)
        expanded += 1

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
                )

            frontier.append(child_node)
            frontier_states.add(child_state)

    return SearchResult(
        algorithm="bfs",
        success=False,
        cost=None,
        path=(),
        expanded_nodes=expanded,
        frontier_nodes=0,
        elapsed_seconds=time.perf_counter() - start,
    )


def dfs(problem: Problem) -> SearchResult:
    """Depth-first search: uninformed graph search expanding depth-first. SRCH-02."""
    start = time.perf_counter()
    root = SearchNode.root(problem.initial)

    frontier: list[SearchNode] = [root]
    explored: set[GameState] = set()
    expanded = 0

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
            )

        if node.state in explored:
            continue

        explored.add(node.state)
        expanded += 1

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

    return SearchResult(
        algorithm="dfs",
        success=False,
        cost=None,
        path=(),
        expanded_nodes=expanded,
        frontier_nodes=0,
        elapsed_seconds=time.perf_counter() - start,
    )


def greedy(problem: Problem, heuristic: Heuristic) -> SearchResult:
    """Greedy best-first search: informed, orders the frontier by heuristic value alone. SRCH-03."""
    start = time.perf_counter()
    root = SearchNode.root(problem.initial)

    # Entry format: (h(state), tie_breaker_counter, node)
    counter = 0
    heap: list[tuple[float, int, SearchNode]] = [(heuristic(problem, root.state), counter, root)]
    explored: set[GameState] = set()
    expanded = 0

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
            )

        if node.state in explored:
            continue

        explored.add(node.state)
        expanded += 1

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

    return SearchResult(
        algorithm="greedy",
        success=False,
        cost=None,
        path=(),
        expanded_nodes=expanded,
        frontier_nodes=0,
        elapsed_seconds=time.perf_counter() - start,
    )


def astar(problem: Problem, heuristic: Heuristic) -> SearchResult:
    """A* search: informed, orders the frontier by path cost plus heuristic value. SRCH-04."""
    start = time.perf_counter()
    root = SearchNode.root(problem.initial)

    best_g: dict[GameState, int] = {root.state: 0}
    counter = 0
    h_root = heuristic(problem, root.state)
    heap: list[tuple[float, int, SearchNode]] = [(h_root, counter, root)]
    expanded = 0

    while heap:
        _, _, node = heapq.heappop(heap)

        # Stale entry check: if a cheaper path to this state was already processed
        if node.path_cost > best_g.get(node.state, float("inf")):
            continue

        if problem.is_goal(node.state):
            return SearchResult(
                algorithm="astar",
                success=True,
                cost=node.path_cost,
                path=node.path(),
                expanded_nodes=expanded,
                frontier_nodes=len(heap),
                elapsed_seconds=time.perf_counter() - start,
            )

        expanded += 1

        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            g = node.path_cost + problem.step_cost(node.state, action, child_state)

            if g < best_g.get(child_state, float("inf")):
                best_g[child_state] = g
                child_node = SearchNode(
                    state=child_state,
                    parent=node,
                    action=action,
                    path_cost=g,
                    depth=node.depth + 1,
                )
                counter += 1
                f = g + heuristic(problem, child_state)
                heapq.heappush(heap, (f, counter, child_node))

    return SearchResult(
        algorithm="astar",
        success=False,
        cost=None,
        path=(),
        expanded_nodes=expanded,
        frontier_nodes=0,
        elapsed_seconds=time.perf_counter() - start,
    )


def iddfs(problem: Problem, max_depth: int = 1000) -> SearchResult:
    """Iterative deepening DFS: uninformed, expands depth-first with increasing limits. SRCH-05."""
    start = time.perf_counter()
    total_expanded = 0

    for limit in range(max_depth + 1):
        cutoff_occurred = False
        # Frontier stack contains (node, path_states_on_branch)
        root = SearchNode.root(problem.initial)
        frontier: list[tuple[SearchNode, frozenset[GameState]]] = [(root, frozenset([root.state]))]

        while frontier:
            node, branch = frontier.pop()

            if problem.is_goal(node.state):
                return SearchResult(
                    algorithm="iddfs",
                    success=True,
                    cost=node.path_cost,
                    path=node.path(),
                    expanded_nodes=total_expanded,
                    frontier_nodes=len(frontier),
                    elapsed_seconds=time.perf_counter() - start,
                )

            if node.depth >= limit:
                cutoff_occurred = True
                continue

            total_expanded += 1

            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                if child_state not in branch:
                    child_node = SearchNode(
                        state=child_state,
                        parent=node,
                        action=action,
                        path_cost=node.path_cost + problem.step_cost(node.state, action, child_state),
                        depth=node.depth + 1,
                    )
                    frontier.append((child_node, branch | {child_state}))

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
    )


ALGORITHMS: dict[str, Callable[..., SearchResult]] = {
    "bfs": bfs,
    "dfs": dfs,
    "greedy": greedy,
    "astar": astar,
    "iddfs": iddfs,
}
