"""Open the real game window and watch ``random_search``'s solution play itself.

Not part of the graded deliverable: a visual demo of the search
scaffolding, reusing the same renderer a human plays with
(``gridworld.ui.render.draw_frame``) so watching it is exactly like
watching a very fast, slightly wasteful player solve the level by hand.
One move from the found path is applied every ``STEP_DELAY_MS``; the win
overlay appears automatically once the replayed state is solved, because
``draw_frame`` already checks that on every frame.

Usage::

    py -3 TP1/scripts/replay_random_search.py [level_path]

Defaults to ``levels/01-warmup.json`` (small board, finishes in well under
a second of search and ~5 seconds of animation). Larger boards can take
``random_search`` much longer to find a solution -- see the demo's own
docstring in ``gridworld/search/demo.py`` for why an unguided algorithm
struggles as the state space grows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure TP1 directory is on sys.path when executed directly as a script
_TP1_DIR = Path(__file__).resolve().parent.parent
if str(_TP1_DIR) not in sys.path:
    sys.path.insert(0, str(_TP1_DIR))

import pygame

from gridworld.engine.rules import apply_move
from gridworld.history import MoveHistory
from gridworld.levelfile import DEFAULT_LEVEL_PATH, LevelError, load_level
from gridworld.search.demo import random_search
from gridworld.search.problem import Problem
from gridworld.ui.render import build_fonts, draw_frame
from gridworld.ui.sprites import build_sprites
from gridworld.ui.theme import WINDOW_SIZE, WINDOW_TITLE, cell_size

STEP_DELAY_MS = 300


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open the real game window and watch random_search's solution play itself."
    )
    parser.add_argument(
        "level_path",
        nargs="?",
        default=None,
        help="Path to the JSON level file (defaults to levels/01-warmup.json)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Random seed (number or string). Pass 'random' or 'none' for a new random solution on every run.",
    )
    args = parser.parse_args()

    level_path = args.level_path if args.level_path is not None else DEFAULT_LEVEL_PATH

    seed_val: int | str | None
    if args.seed is None:
        seed_val = str(level_path)
    elif args.seed.lower() in ("random", "none"):
        seed_val = None
    elif args.seed.isdigit() or (args.seed.startswith("-") and args.seed[1:].isdigit()):
        seed_val = int(args.seed)
    else:
        seed_val = args.seed

    try:
        level = load_level(level_path)
    except LevelError as err:
        sys.stderr.write(f"{err}\n")
        return 1

    problem = Problem(board=level.board, initial=level.state)
    result = random_search(problem, seed=seed_val)
    if not result.success:
        sys.stderr.write(
            f"random_search did not find a solution for {level_path} "
            f"within its expansion budget\n"
        )
        return 1

    print(
        f"{level.name}: random_search found a {result.cost}-move solution "
        f"({result.expanded_nodes} nodes expanded, "
        f"{result.elapsed_seconds:.3f}s to search) -- now replaying it"
    )

    history = MoveHistory.start(level.state)
    remaining = list(result.path)
    selected: int | None = None
    announced_solved = False

    pygame.init()
    pygame.display.set_caption(f"{WINDOW_TITLE} — random_search replay")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    fonts = build_fonts()
    sprites = build_sprites(level.board, cell_size(level.board.cols, level.board.rows))

    next_step_ms = pygame.time.get_ticks() + STEP_DELAY_MS
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        now = pygame.time.get_ticks()
        if remaining and now >= next_step_ms:
            action = remaining.pop(0)
            outcome = apply_move(level.board, history.current, action.car, action.direction)
            assert outcome.accepted, (action.car, action.direction, outcome.rejection)
            history = history.push(outcome.state)
            selected = None if outcome.state.is_parked(action.car) else action.car
            next_step_ms = now + STEP_DELAY_MS

        if not remaining and not announced_solved:
            print("Solved -- close the window (Esc) when you're done watching.")
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
