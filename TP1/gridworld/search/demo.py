"""A throwaway algorithm to exercise the search scaffolding end to end.

``random_search`` is **not** one of SRCH-01..05 and is not a heuristic --
it exists only to prove ``Problem``, ``SearchNode``, and ``SearchResult``
actually work together before the real algorithms in ``algorithms.py``
are written. Do not submit this as the TP's BFS/DFS/Greedy/A*/IDDFS.

It is still a complete graph search, not a blind random walk: the
frontier holds every generated-but-not-yet-expanded node, a visited set
of ``GameState`` (hashable, per the engine's contract) prevents
re-expanding a state reached twice, and only the order nodes are pulled
off the frontier is random rather than by depth, cost, or heuristic. So
it terminates with ``success=True`` whenever a solution exists, the same
way an unordered BFS would -- it is just slower and finds a longer path.
"""

from __future__ import annotations

import random
import time

from gridworld.engine.state import GameState
from gridworld.search.metrics import SearchResult
from gridworld.search.node import SearchNode
from gridworld.search.problem import Problem


def random_search(
    problem: Problem,
    *,
    seed: int | str | None = None,
    max_expansions: int = 20_000,
) -> SearchResult:
    """Expand nodes off the frontier in uniform-random order until a goal is found.

    Returns a ``SearchResult`` with ``success=False`` if the frontier is
    exhausted, or ``max_expansions`` nodes are expanded, before a goal
    state is reached.
    """
    rng = random.Random(seed)
    start = time.perf_counter()

    root = SearchNode.root(problem.initial)
    frontier: list[SearchNode] = [root]
    visited: set[GameState] = {root.state}
    expanded = 0

    while frontier:
        node = frontier.pop(rng.randrange(len(frontier)))

        if problem.is_goal(node.state):
            return SearchResult(
                algorithm="random",
                success=True,
                cost=node.path_cost,
                path=node.path(),
                expanded_nodes=expanded,
                frontier_nodes=len(frontier),
                elapsed_seconds=time.perf_counter() - start,
            )

        if expanded >= max_expansions:
            break
        expanded += 1

        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if child_state in visited:
                continue
            visited.add(child_state)
            frontier.append(
                SearchNode(
                    state=child_state,
                    parent=node,
                    action=action,
                    path_cost=node.path_cost + problem.step_cost(node.state, action, child_state),
                    depth=node.depth + 1,
                )
            )

    return SearchResult(
        algorithm="random",
        success=False,
        cost=None,
        path=(),
        expanded_nodes=expanded,
        frontier_nodes=len(frontier),
        elapsed_seconds=time.perf_counter() - start,
    )
