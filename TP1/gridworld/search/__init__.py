"""Search engine for Grid World.

Sibling to ``gridworld.engine``, under the same constraint: nothing here
may import pygame or any other rendering library at any depth, and this
package must import and run headlessly. A* and its first admissible heuristic
power the in-game optimal-search animation; the remaining entries stay wired
as later-milestone stubs.
"""

from gridworld.search.algorithms import (
    ALGORITHMS,
    AStarStepper,
    SearchExpansion,
    astar,
    bfs,
    dfs,
    greedy,
    iddfs,
)
from gridworld.search.heuristics import HEURISTICS, Heuristic, heuristic_a, heuristic_b
from gridworld.search.metrics import SearchResult
from gridworld.search.node import SearchNode
from gridworld.search.problem import Problem

__all__ = [
    "Problem",
    "SearchNode",
    "SearchResult",
    "Heuristic",
    "heuristic_a",
    "heuristic_b",
    "HEURISTICS",
    "bfs",
    "dfs",
    "greedy",
    "astar",
    "iddfs",
    "ALGORITHMS",
    "AStarStepper",
    "SearchExpansion",
]
