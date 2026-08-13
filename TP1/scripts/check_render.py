"""Prove UI requirements on real pixels under the dummy video driver.

This is the renderer smoke check: it runs under pygame's dummy video
driver so it needs no display. Decision D-04 keeps the pytest suite
renderer-free (no test imports ``gridworld.ui``, constructs a surface, or
opens a display), so renderer behaviour needs a script rather than a test.
This script is the automated evidence for the Phase 3 UI requirements;
Phase 4 re-runs it at delivery.

Run it with the WSL venv interpreter (the one with pygame installed)::

    wsl -d Ubuntu-24.04 --cd "<repo>/TP1" \\
        -e /home/lucasdicandia/.venvs/gridworld/bin/python scripts/check_render.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

TP1_DIR = Path(__file__).resolve().parent.parent
if str(TP1_DIR) not in sys.path:
    sys.path.insert(0, str(TP1_DIR))

import pygame  # noqa: E402

from gridworld.levelfile import load_level  # noqa: E402
from gridworld.ui.render import build_fonts, draw_frame  # noqa: E402
from gridworld.ui.sprites import build_sprites  # noqa: E402
from gridworld.ui.theme import (  # noqa: E402
    COLOR_BOARD_BG,
    COLOR_OBSTACLE,
    WINDOW_SIZE,
    cell_size,
    grid_origin,
)


def _cell_center(board_cols: int, board_rows: int, row: int, col: int) -> tuple[int, int]:
    cell = cell_size(board_cols, board_rows)
    origin_x, origin_y = grid_origin(board_cols, board_rows)
    return (origin_x + col * cell + cell // 2, origin_y + row * cell + cell // 2)


def check_legal_destinations() -> bool:
    """Selecting car 1 on the warmup level tints exactly its two destinations."""
    pygame.init()
    surface = pygame.display.set_mode(WINDOW_SIZE)

    level = load_level("levels/01-warmup.json")
    board, state = level.board, level.state
    fonts = build_fonts()
    sprites = build_sprites(board, cell_size(board.cols, board.rows))

    draw_frame(surface, fonts, board, state, selected=1, moves=0, sprites=sprites)

    center_1_0 = _cell_center(board.cols, board.rows, 1, 0)
    center_0_1 = _cell_center(board.cols, board.rows, 0, 1)
    center_1_1 = _cell_center(board.cols, board.rows, 1, 1)
    center_2_1 = _cell_center(board.cols, board.rows, 2, 1)

    ok = True
    ok &= surface.get_at(center_1_0)[:3] != COLOR_BOARD_BG
    ok &= surface.get_at(center_0_1)[:3] != COLOR_BOARD_BG
    ok &= surface.get_at(center_1_1)[:3] == COLOR_BOARD_BG
    ok &= surface.get_at(center_2_1)[:3] == COLOR_OBSTACLE

    draw_frame(surface, fonts, board, state, selected=None, moves=0, sprites=sprites)
    ok &= surface.get_at(center_1_0)[:3] == COLOR_BOARD_BG

    return bool(ok)


def _report(label: str, ok: bool) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"{status}  {label}")
    return ok


def main() -> int:
    checks = (
        ("selecting car 1 tints exactly its legal destinations", check_legal_destinations),
    )

    passed = 0
    for label, check in checks:
        if _report(label, check()):
            passed += 1

    print(f"\n{passed}/{len(checks)} render checks passing")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
