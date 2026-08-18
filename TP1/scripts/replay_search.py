"""Visual runner and replayer for all search algorithms (BFS, DFS, Greedy, A*, IDDFS).

Usage:
    py scripts/replay_search.py --algo <bfs|dfs|greedy|astar|iddfs> [level_path] [--heuristic <heuristic_a|heuristic_b>] [--delay <ms>]

Examples:
    py scripts/replay_search.py --algo bfs levels/01-warmup.json
    py scripts/replay_search.py --algo astar levels/02-classic.json --heuristic heuristic_a
    py scripts/replay_search.py --algo greedy levels/02-classic.json
    py scripts/replay_search.py --algo dfs levels/01-warmup.json --delay 150
"""

from __future__ import annotations

import argparse
from collections import deque
from enum import Enum
from pathlib import Path
import sys

# Ensure TP1 directory is on sys.path when executed directly as a script
_TP1_DIR = Path(__file__).resolve().parent.parent
if str(_TP1_DIR) not in sys.path:
    sys.path.insert(0, str(_TP1_DIR))

import pygame

from gridworld.engine.board import Position
from gridworld.engine.rules import apply_move
from gridworld.engine.state import GameState
from gridworld.history import MoveHistory
from gridworld.levelfile import DEFAULT_LEVEL_PATH, LevelError, load_level
from gridworld.search.algorithms import ALGORITHMS
from gridworld.search.heuristics import HEURISTICS
from gridworld.search.problem import Problem
from gridworld.ui.app import build_solution_paths
from gridworld.ui.render import build_fonts, draw_frame
from gridworld.ui.sprites import build_sprites
from gridworld.ui.theme import (
    SEARCH_REVEAL_DELAYS_MS,
    SEARCH_TRAIL_REVEAL_MS,
    WINDOW_SIZE,
    WINDOW_TITLE,
    cell_size,
)


class ReplayPhase(Enum):
    """Visible phases of the offline replay: explore, trace, then move."""

    EXPLORING = "exploring"
    TRAIL_PAUSE = "trail_pause"
    REPLAYING = "replaying"
    DONE = "done"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a search algorithm on a level and visually replay its solution in Pygame."
    )
    parser.add_argument(
        "level_path",
        nargs="?",
        default=None,
        help="Path to the JSON level file (defaults to levels/01-warmup.json)",
    )
    parser.add_argument(
        "--algo",
        "-a",
        type=str,
        default="astar",
        choices=list(ALGORITHMS.keys()),
        help=f"Search algorithm to use: {', '.join(ALGORITHMS.keys())} (default: astar)",
    )
    parser.add_argument(
        "--heuristic",
        "-he",
        type=str,
        default="heuristic_a",
        choices=list(HEURISTICS.keys()),
        help=f"Heuristic function for Greedy and A*: {', '.join(HEURISTICS.keys())} (default: heuristic_a)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=int,
        default=250,
        help="Milliseconds delay between replayed steps in animation (default: 250)",
    )
    args = parser.parse_args()

    level_path = args.level_path if args.level_path is not None else DEFAULT_LEVEL_PATH

    try:
        level = load_level(level_path)
    except LevelError as err:
        sys.stderr.write(f"Error loading level: {err}\n")
        return 1

    problem = Problem(board=level.board, initial=level.state)
    algo_fn = ALGORITHMS[args.algo]

    # Every state the algorithm expands is recorded here, per car, in the
    # order it was expanded -- the reveal animation just replays this list.
    expansion_order: list[tuple[int, Position]] = []
    known_cells: dict[int, set[Position]] = {}

    def _record_expansion(state: GameState) -> None:
        for car, position in enumerate(state.cars, start=1):
            car_known = known_cells.setdefault(car, set())
            if position not in car_known:
                car_known.add(position)
                expansion_order.append((car, position))

    print("=" * 60)
    print(f"Running {args.algo.upper()} on level '{level.name}' ({level_path})...")
    if args.algo in ("greedy", "astar"):
        print(f"Heuristic: {args.heuristic}")
        heuristic_fn = HEURISTICS[args.heuristic]
        result = algo_fn(problem, heuristic_fn, on_expand=_record_expansion)
    else:
        result = algo_fn(problem, on_expand=_record_expansion)

    print("=" * 60)
    print("SEARCH RESULTS:")
    print(f"  • Success:         {result.success}")
    print(f"  • Cost:            {result.cost}")
    print(f"  • Path Length:     {len(result.path)} moves")
    print(f"  • Expanded Nodes:  {result.expanded_nodes}")
    print(f"  • Frontier Nodes:  {result.frontier_nodes}")
    print(f"  • Elapsed Time:    {result.elapsed_seconds * 1000:.2f} ms ({result.elapsed_seconds:.4f} s)")
    print("=" * 60)

    if not result.success:
        sys.stderr.write(f"Algorithm {args.algo} did not find a solution for {level_path}.\n")
        return 1

    full_solution_paths = build_solution_paths(level.board, level.state, result.path)

    print("Starting visual replay window...")
    print("Controls in replay:")
    print("  • Space: Pause / Resume replay")
    print("  • Esc: Close window")

    history = MoveHistory.start(level.state)
    remaining_path = deque(result.path)
    pending_cells = deque(expansion_order)
    explored_cells: dict[int, set[Position]] = {}
    solution_paths: dict[int, tuple[Position, ...]] = {}
    focus_cell: tuple[int, Position] | None = None
    selected: int | None = None
    announced_solved = False
    paused = False
    phase = ReplayPhase.EXPLORING if pending_cells else ReplayPhase.TRAIL_PAUSE

    pygame.init()
    pygame.display.set_caption(f"{WINDOW_TITLE} — {args.algo.upper()} Replay ({level.name})")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    fonts = build_fonts()
    sprites = build_sprites(level.board, cell_size(level.board.cols, level.board.rows))

    next_visual_ms = pygame.time.get_ticks()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    next_visual_ms = pygame.time.get_ticks()

        now = pygame.time.get_ticks()

        if not paused and phase is ReplayPhase.EXPLORING and now >= next_visual_ms:
            car, position = pending_cells.popleft()
            explored_cells.setdefault(car, set()).add(position)
            focus_cell = (car, position)
            next_visual_ms = now + SEARCH_REVEAL_DELAYS_MS[1]
            if not pending_cells:
                focus_cell = None
                phase = ReplayPhase.TRAIL_PAUSE
                next_visual_ms = now + SEARCH_TRAIL_REVEAL_MS

        if not paused and phase is ReplayPhase.TRAIL_PAUSE and now >= next_visual_ms:
            solution_paths = {car: tuple(path) for car, path in full_solution_paths.items()}
            phase = ReplayPhase.REPLAYING
            next_visual_ms = now

        if not paused and phase is ReplayPhase.REPLAYING and now >= next_visual_ms:
            if not remaining_path:
                phase = ReplayPhase.DONE
            else:
                action = remaining_path.popleft()
                outcome = apply_move(level.board, history.current, action.car, action.direction)
                assert outcome.accepted, (action.car, action.direction, outcome.rejection)
                history = history.push(outcome.state)
                selected = None if outcome.state.is_parked(action.car) else action.car
                next_visual_ms = now + args.delay
                if not remaining_path:
                    phase = ReplayPhase.DONE

        if phase is ReplayPhase.DONE and not announced_solved:
            print("Level solved! Press Esc to exit.")
            announced_solved = True

        draw_frame(
            screen,
            fonts,
            level.board,
            history.current,
            selected,
            history.depth,
            sprites,
            explored_cells={car: frozenset(cells) for car, cells in explored_cells.items()},
            solution_paths=solution_paths,
            focus_cell=focus_cell,
        )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
