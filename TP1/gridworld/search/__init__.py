"""Search-engine scaffolding for Grid World (v2 milestone).

Sibling to ``gridworld.engine``, under the same constraint: nothing here
may import pygame or any other rendering library at any depth, and this
package must import and run headlessly. Every algorithm and heuristic in
this package is a stub -- ``ALGORITHMS`` and ``HEURISTICS`` are wired and
ready, but raise ``NotImplementedError`` until their bodies are written.
"""

from gridworld.search.algorithms import ALGORITHMS, astar, bfs, dfs, greedy, iddfs
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
]
