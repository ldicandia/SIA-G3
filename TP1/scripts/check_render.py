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

from gridworld.engine.board import Board  # noqa: E402
from gridworld.engine.rules import stranded_cars  # noqa: E402
from gridworld.engine.state import GameState  # noqa: E402
from gridworld.levelfile import load_level  # noqa: E402
from gridworld.ui.app import Session, _handle_keydown, _start_level  # noqa: E402
from gridworld.ui.render import build_fonts, draw_frame  # noqa: E402
from gridworld.ui.sprites import build_sprites  # noqa: E402
from gridworld.ui.theme import (  # noqa: E402
    BOARD_SIZE,
    COLOR_BOARD_BG,
    COLOR_DESTRUCTIVE,
    COLOR_HUD_BG,
    COLOR_OBSTACLE,
    CONTROL_LEGEND,
    FONT_BODY,
    HUD_WARNING_BLOCK_HEIGHT,
    HUD_WIDTH,
    SPACING_LG,
    SPACING_MD,
    SPACING_XS,
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


def check_session_key_handling() -> bool:
    """Test Session state machine and keydown transitions headlessly."""
    pygame.init()
    pygame.display.set_mode(WINDOW_SIZE)

    level = load_level("levels/01-warmup.json")
    session = Session()
    _start_level(session, level)

    # 1. Selecting car 1
    _handle_keydown(session, pygame.K_1)
    if session.selected != 1:
        return False

    # 2. Drive car 1 down 3, right 4, down 1 to land on flag (4, 4)
    moves = [
        pygame.K_DOWN, pygame.K_DOWN, pygame.K_DOWN,
        pygame.K_RIGHT, pygame.K_RIGHT, pygame.K_RIGHT, pygame.K_RIGHT,
        pygame.K_DOWN,
    ]
    for key in moves:
        _handle_keydown(session, key)

    # Selection must be automatically cleared upon parking car 1 (WR-01)
    if session.selected is not None:
        return False
    if not session.history.current.is_parked(1):
        return False

    # 3. Selecting a parked car flashes its position and rejects selection
    _handle_keydown(session, pygame.K_1)
    if session.selected is not None:
        return False
    if session.flash_cell != (4, 4):
        return False

    # 4. Arrow key with no selection does nothing
    _handle_keydown(session, pygame.K_LEFT)
    if session.history.depth != 8:
        return False

    # 5. Undo (K_u) un-parks car 1 and restores depth
    _handle_keydown(session, pygame.K_u)
    if session.history.current.is_parked(1):
        return False
    if session.history.depth != 7:
        return False

    # 6. Reset (K_r) clears selection and resets history depth to 0
    _handle_keydown(session, pygame.K_r)
    if session.history.depth != 0:
        return False
    if session.selected is not None:
        return False

    return True


def check_control_legend() -> bool:
    """The legend is five rows, row 2 is the undo row, and every row is
    typesettable by the bundled font (ASCII plus the em dash and middle
    dot only -- anything else renders as a ``.notdef`` box). No display
    reads needed.
    """
    if len(CONTROL_LEGEND) != 5:
        return False
    if "undo" not in CONTROL_LEGEND[2]:
        return False
    if "reset" not in CONTROL_LEGEND[3]:
        return False
    if "quit" not in CONTROL_LEGEND[4]:
        return False

    allowed_extra = {"—", "·"}
    for row in CONTROL_LEGEND:
        for char in row:
            if ord(char) >= 128 and char not in allowed_extra:
                return False
    return True


def _warning_block_origin(fonts) -> tuple[int, int]:
    """Reproduce draw_hud's own cursor arithmetic to find the reserved block's origin.

    Duplicated (not imported) for the same reason ``check_hud_fits`` duplicates
    it: a real assertion against real font metrics under the dummy driver,
    not a hardcoded pixel guess.
    """
    x = BOARD_SIZE + SPACING_LG
    y = SPACING_LG
    for _ in range(2):  # Moves block, then Car block.
        y += fonts.label.get_height() + SPACING_MD
        y += fonts.display.get_height() + SPACING_LG
    y += fonts.label.get_height() + SPACING_MD  # Controls label.
    line_height = round(FONT_BODY * 1.4)
    y += len(CONTROL_LEGEND) * line_height
    y += SPACING_LG
    return x, y


def check_unwinnable_warning() -> bool:
    """The reserved HUD block stays blank when suppressed, fills when enabled.

    Builds the 3x3 stranded board (car 1 boxed in), draws one frame with
    the warning suppressed and one with it enabled, and samples pixels
    across the reserved block region. Also proves -- on real font metrics
    under the dummy driver, not an assumption -- that the composed warning
    height never exceeds HUD_WARNING_BLOCK_HEIGHT, the exact Phase 1
    reservation this plan fills without growing.
    """
    pygame.init()
    surface = pygame.display.set_mode(WINDOW_SIZE)

    board = Board(
        rows=3, cols=3, obstacles=frozenset({(0, 1), (1, 0)}), flags=((2, 2), (2, 1))
    )
    state = GameState(cars=((0, 0), (2, 0)), parked=frozenset())
    stranded = stranded_cars(board, state)
    assert stranded == (1,), stranded

    fonts = build_fonts()
    sprites = build_sprites(board, cell_size(board.cols, board.rows))

    x, y = _warning_block_origin(fonts)
    region_width = HUD_WIDTH - 2 * SPACING_LG
    probe_points = [
        (px, py)
        for py in range(y, y + HUD_WARNING_BLOCK_HEIGHT, 2)
        for px in range(x, x + region_width, 8)
    ]

    draw_frame(surface, fonts, board, state, selected=None, moves=0, sprites=sprites, unwinnable=False, stranded=())
    suppressed_ok = all(surface.get_at(p)[:3] == COLOR_HUD_BG for p in probe_points)

    draw_frame(surface, fonts, board, state, selected=None, moves=0, sprites=sprites, unwinnable=True, stranded=stranded)
    enabled_ok = any(surface.get_at(p)[:3] == COLOR_DESTRUCTIVE for p in probe_points)

    heading_label = fonts.label.render("Unwinnable", True, COLOR_DESTRUCTIVE)
    body_label = fonts.body.render("placeholder", True, COLOR_DESTRUCTIVE)
    hint_label = fonts.body.render("placeholder", True, COLOR_HUD_BG)
    full_height = (
        heading_label.get_height()
        + SPACING_XS
        + body_label.get_height()
        + SPACING_XS
        + hint_label.get_height()
    )
    composed_height = full_height if full_height <= HUD_WARNING_BLOCK_HEIGHT else (
        body_label.get_height() + SPACING_XS + hint_label.get_height()
    )
    height_ok = composed_height <= HUD_WARNING_BLOCK_HEIGHT

    return bool(suppressed_ok and enabled_ok and height_ok)


def check_hud_fits() -> bool:
    """The HUD stack -- five-row legend included -- still fits above the
    reserved warning block, under the dummy video driver.

    Deliberately duplicates a small slice of ``draw_hud``'s layout
    arithmetic (top padding, then for each of the Moves, Car and Controls
    blocks the label height plus the inter-label gap plus the value height
    plus the inter-block gap, with the legend contributing
    ``len(CONTROL_LEGEND) * round(FONT_BODY * 1.4)``), present so that
    adding this plan's fifth legend row cannot silently push 03-03's
    warning block off the panel.
    """
    pygame.init()
    pygame.display.set_mode(WINDOW_SIZE)
    fonts = build_fonts()

    y = SPACING_LG
    for _ in range(2):  # Moves block, then Car block.
        y += fonts.label.get_height() + SPACING_MD
        y += fonts.display.get_height() + SPACING_LG

    y += fonts.label.get_height() + SPACING_MD  # Controls label.
    line_height = round(FONT_BODY * 1.4)
    y += len(CONTROL_LEGEND) * line_height
    y += SPACING_LG

    stack_height = y + HUD_WARNING_BLOCK_HEIGHT
    return stack_height <= WINDOW_SIZE[1] - SPACING_LG


def _report(label: str, ok: bool) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"{status}  {label}")
    return ok


def main() -> int:
    checks = (
        ("selecting car 1 tints exactly its legal destinations", check_legal_destinations),
        ("session key handling state machine behavior", check_session_key_handling),
        ("control legend is five rows with undo inserted, typesettable", check_control_legend),
        ("HUD stack fits alongside the reserved warning block", check_hud_fits),
        ("unwinnable warning fills the reserved block without growing it", check_unwinnable_warning),
    )

    passed = 0
    for label, check in checks:
        if _report(label, check()):
            passed += 1

    print(f"\n{passed}/{len(checks)} render checks passing")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
