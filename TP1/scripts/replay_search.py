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
from pathlib import Path
import sys

# Ensure TP1 directory is on sys.path when executed directly as a script
_TP1_DIR = Path(__file__).resolve().parent.parent
if str(_TP1_DIR) not in sys.path:
    sys.path.insert(0, str(_TP1_DIR))

import pygame

from gridworld.engine.rules import apply_move
from gridworld.history import MoveHistory
from gridworld.levelfile import DEFAULT_LEVEL_PATH, LevelError, load_level
from gridworld.search.algorithms import ALGORITHMS
from gridworld.search.heuristics import HEURISTICS
from gridworld.search.problem import Problem
from gridworld.ui.render import build_fonts, draw_frame
from gridworld.ui.sprites import build_sprites
from gridworld.ui.theme import WINDOW_SIZE, WINDOW_TITLE, cell_size


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

    print("=" * 60)
    print(f"Running {args.algo.upper()} on level '{level.name}' ({level_path})...")
    if args.algo in ("greedy", "astar"):
        print(f"Heuristic: {args.heuristic}")
        heuristic_fn = HEURISTICS[args.heuristic]
        result = algo_fn(problem, heuristic_fn)
    else:
        result = algo_fn(problem)

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

    print("Starting visual replay window...")
    print("Controls in replay:")
    print("  • Space: Pause / Resume replay")
    print("  • Esc: Close window")

    history = MoveHistory.start(level.state)
    remaining = list(result.path)
    selected: int | None = None
    announced_solved = False
    paused = False

    pygame.init()
    pygame.display.set_caption(f"{WINDOW_TITLE} — {args.algo.upper()} Replay ({level.name})")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    fonts = build_fonts()
    sprites = build_sprites(level.board, cell_size(level.board.cols, level.board.rows))

    next_step_ms = pygame.time.get_ticks() + args.delay
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

        now = pygame.time.get_ticks()
        if not paused and remaining and now >= next_step_ms:
            action = remaining.pop(0)
            outcome = apply_move(level.board, history.current, action.car, action.direction)
            assert outcome.accepted, (action.car, action.direction, outcome.rejection)
            history = history.push(outcome.state)
            selected = None if outcome.state.is_parked(action.car) else action.car
            next_step_ms = now + args.delay

        if not remaining and not announced_solved:
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
        )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
