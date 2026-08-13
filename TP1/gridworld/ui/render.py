"""Board state treatments, HUD panel, illegal-move flash and win overlay.

Every colour, spacing value, font size and on-screen string is read from
``gridworld.ui.theme`` -- nothing here is written inline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from gridworld.engine.board import Board
from gridworld.engine.rules import is_solved
from gridworld.engine.state import GameState
from gridworld.levelfile import LevelEntry
from gridworld.ui.sprites import SpriteSet, arrow_row_surface
from gridworld.ui.theme import (
    BOARD_SIZE,
    CONTROL_LEGEND,
    CONTROL_LEGEND_ARROW_ROW,
    COLOR_BOARD_BG,
    COLOR_DESTRUCTIVE,
    COLOR_GRID_LINE,
    COLOR_HUD_BG,
    COLOR_OBSTACLE,
    COLOR_SCRIM,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    FLASH_ALPHA,
    FONT_BODY,
    FONT_DISPLAY,
    FONT_HEADING,
    FONT_LABEL,
    GRID_LINE_WIDTH,
    HUD_WARNING_BLOCK_HEIGHT,
    HUD_WIDTH,
    LABEL_CAR,
    LABEL_CONTROLS,
    LABEL_MOVES,
    LOAD_ERROR_HEADING,
    LOAD_ERROR_HINT,
    LOAD_ERROR_MAX_LINES,
    NO_CAR_SELECTED,
    PARKED_FILL_ALPHA,
    PICKER_EMPTY,
    PICKER_HEADING,
    PICKER_HINT,
    PICKER_MAX_CHOICES,
    PICKER_NAME_MAX_CHARS,
    PICKER_UNSELECTABLE,
    SCRIM_ALPHA,
    SELECTION_OUTLINE_WIDTH,
    SPACING_2XL,
    SPACING_3XL,
    SPACING_LG,
    SPACING_MD,
    SPACING_SM,
    SPACING_XS,
    WIN_BODY,
    WINDOW_SIZE,
    car_hue,
    cell_size,
    grid_origin,
    picker_entry_detail,
    win_heading,
)


@dataclass(frozen=True)
class Fonts:
    """The four typography roles, built once at startup, never per frame."""

    body: pygame.font.Font
    label: pygame.font.Font
    heading: pygame.font.Font
    display: pygame.font.Font


def build_fonts() -> Fonts:
    """Build the four theme fonts once. Call at startup, not per frame."""
    default = pygame.font.get_default_font()
    body = pygame.font.Font(default, FONT_BODY)
    label = pygame.font.Font(default, FONT_LABEL)
    heading = pygame.font.Font(default, FONT_HEADING)
    display = pygame.font.Font(default, FONT_DISPLAY)
    label.set_bold(True)
    heading.set_bold(True)
    display.set_bold(True)
    return Fonts(body=body, label=label, heading=heading, display=display)


def _blit_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
) -> pygame.Rect:
    label = font.render(text, True, color)
    rect = label.get_rect(topleft=pos)
    surface.blit(label, rect)
    return rect


def draw_board(
    surface: pygame.Surface,
    board: Board,
    state: GameState,
    selected: int | None,
    sprites: SpriteSet,
) -> None:
    """Render every cell of ``board`` exactly once, in State Treatment order.

    Empty and obstacle cells first, then unclaimed flags, then cars (idle,
    parked, selected). Drawing is clipped to the board area so a grid
    clamped by ``MIN_CELL`` past the board area is cut off at the edge
    rather than bleeding into the HUD panel.
    """
    cell = cell_size(board.cols, board.rows)
    origin_x, origin_y = grid_origin(board.cols, board.rows)
    board_rect = pygame.Rect(0, 0, BOARD_SIZE, BOARD_SIZE)

    surface.fill(COLOR_BOARD_BG, board_rect)
    previous_clip = surface.get_clip()
    surface.set_clip(board_rect)

    for row in range(board.rows):
        for col in range(board.cols):
            rect = pygame.Rect(origin_x + col * cell, origin_y + row * cell, cell, cell)
            if board.is_obstacle((row, col)):
                pygame.draw.rect(surface, COLOR_OBSTACLE, rect)
            else:
                pygame.draw.rect(surface, COLOR_BOARD_BG, rect)
                pygame.draw.rect(surface, COLOR_GRID_LINE, rect, width=GRID_LINE_WIDTH)

    for car in board.car_numbers():
        if car in state.parked:
            continue
        row, col = board.flag_for(car)
        rect = pygame.Rect(origin_x + col * cell, origin_y + row * cell, cell, cell)
        surface.blit(sprites.flags[car], rect.topleft)

    for car in board.car_numbers():
        row, col = state.position_of(car)
        rect = pygame.Rect(origin_x + col * cell, origin_y + row * cell, cell, cell)
        if car in state.parked:
            hue = car_hue(car)
            fill = pygame.Surface((cell, cell), pygame.SRCALPHA)
            fill.fill((*hue, PARKED_FILL_ALPHA))
            surface.blit(fill, rect.topleft)
            pygame.draw.rect(surface, hue, rect, width=SELECTION_OUTLINE_WIDTH)
            surface.blit(sprites.cars[car], rect.topleft)
        else:
            surface.blit(sprites.cars[car], rect.topleft)
            if selected == car:
                pygame.draw.rect(surface, COLOR_TEXT, rect, width=SELECTION_OUTLINE_WIDTH)

    surface.set_clip(previous_clip)


def draw_hud(surface: pygame.Surface, fonts: Fonts, moves: int, selected: int | None) -> None:
    """Fill the HUD panel and stack every block, every frame, in fixed order."""
    panel_rect = pygame.Rect(BOARD_SIZE, 0, HUD_WIDTH, WINDOW_SIZE[1])
    surface.fill(COLOR_HUD_BG, panel_rect)

    x = BOARD_SIZE + SPACING_LG
    y = SPACING_LG

    _blit_text(surface, fonts.label, LABEL_MOVES, (x, y), COLOR_TEXT)
    y += fonts.label.get_height() + SPACING_MD
    _blit_text(surface, fonts.display, str(moves), (x, y), COLOR_TEXT)
    y += fonts.display.get_height() + SPACING_LG

    _blit_text(surface, fonts.label, LABEL_CAR, (x, y), COLOR_TEXT)
    y += fonts.label.get_height() + SPACING_MD
    car_text = str(selected) if selected is not None else NO_CAR_SELECTED
    _blit_text(surface, fonts.display, car_text, (x, y), COLOR_TEXT)
    y += fonts.display.get_height() + SPACING_LG

    _blit_text(surface, fonts.label, LABEL_CONTROLS, (x, y), COLOR_TEXT)
    y += fonts.label.get_height() + SPACING_MD
    line_height = round(FONT_BODY * 1.4)
    for index, line in enumerate(CONTROL_LEGEND):
        line_x = x
        if index == CONTROL_LEGEND_ARROW_ROW:
            arrows = arrow_row_surface(FONT_BODY, SPACING_XS, COLOR_TEXT)
            arrow_y = y + (fonts.body.get_height() - arrows.get_height()) // 2
            surface.blit(arrows, (line_x, arrow_y))
            line_x += arrows.get_width() + SPACING_SM
        _blit_text(surface, fonts.body, line, (line_x, y), COLOR_TEXT)
        y += line_height
    y += SPACING_LG

    # Reserved, deliberately blank: Phase 3's unwinnable warning lands here
    # without reflowing anything above it.
    _ = pygame.Rect(x, y, HUD_WIDTH - 2 * SPACING_LG, HUD_WARNING_BLOCK_HEIGHT)


def draw_illegal_flash(surface: pygame.Surface, cell_rect: pygame.Rect) -> None:
    """Fill ``cell_rect`` with the destructive colour at flash alpha.

    No text, no state change. Clipped to the board area so an off-grid
    target near the edge cannot bleed into the HUD panel.
    """
    board_rect = pygame.Rect(0, 0, BOARD_SIZE, BOARD_SIZE)
    clipped = cell_rect.clip(board_rect)
    if clipped.width <= 0 or clipped.height <= 0:
        return
    flash = pygame.Surface((clipped.width, clipped.height), pygame.SRCALPHA)
    flash.fill((*COLOR_DESTRUCTIVE, FLASH_ALPHA))
    surface.blit(flash, clipped.topleft)


def draw_win_overlay(surface: pygame.Surface, fonts: Fonts, moves: int) -> None:
    """Draw the scrim and centered win copy over the board area only."""
    board_rect = pygame.Rect(0, 0, BOARD_SIZE, BOARD_SIZE)
    scrim = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
    scrim.fill((*COLOR_SCRIM, SCRIM_ALPHA))
    surface.blit(scrim, board_rect.topleft)

    heading_label = fonts.heading.render(win_heading(moves), True, COLOR_BOARD_BG)
    body_label = fonts.body.render(WIN_BODY, True, COLOR_BOARD_BG)

    total_height = heading_label.get_height() + SPACING_2XL + body_label.get_height()
    center_x = BOARD_SIZE // 2
    top = (BOARD_SIZE - total_height) // 2
    top = max(top, SPACING_3XL)

    heading_rect = heading_label.get_rect(centerx=center_x, top=top)
    surface.blit(heading_label, heading_rect)

    body_rect = body_label.get_rect(centerx=center_x, top=heading_rect.bottom + SPACING_2XL)
    surface.blit(body_label, body_rect)


def draw_frame(
    surface: pygame.Surface,
    fonts: Fonts,
    board: Board,
    state: GameState,
    selected: int | None,
    moves: int,
    sprites: SpriteSet,
    flash_rect: pygame.Rect | None = None,
) -> None:
    """Compose a full frame: board, flash if active, HUD, win overlay if solved."""
    draw_board(surface, board, state, selected, sprites)
    if flash_rect is not None:
        draw_illegal_flash(surface, flash_rect)
    draw_hud(surface, fonts, moves, selected)
    if is_solved(board, state):
        draw_win_overlay(surface, fonts, moves)


def _wrap_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    """Break text into lines splitting on spaces, never exceeding max_width."""
    words = text.split(" ")
    lines: list[str] = []
    current_line: list[str] = []
    for word in words:
        test_line = " ".join(current_line + [word])
        if font.size(test_line)[0] <= max_width or not current_line:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def draw_picker(
    surface: pygame.Surface,
    fonts: Fonts,
    entries: tuple[LevelEntry, ...] | list[LevelEntry],
) -> None:
    """Render the level choice screen composed entirely of theme tokens."""
    surface.fill(COLOR_BOARD_BG)
    _blit_text(surface, fonts.heading, PICKER_HEADING, (SPACING_3XL, SPACING_3XL), COLOR_TEXT)

    if not entries:
        y_empty = SPACING_3XL + fonts.heading.get_height() + SPACING_2XL
        _blit_text(surface, fonts.body, PICKER_EMPTY, (SPACING_3XL, y_empty), COLOR_TEXT_MUTED)
    else:
        y = SPACING_3XL + fonts.heading.get_height() + SPACING_2XL
        row_gap = fonts.body.get_height() + SPACING_MD
        x_num = SPACING_3XL
        x_name = x_num + SPACING_2XL
        x_detail = x_name + 320

        for idx, entry in enumerate(entries):
            num_str = f"{idx + 1}." if idx < PICKER_MAX_CHOICES else PICKER_UNSELECTABLE
            display_name = entry.name[:PICKER_NAME_MAX_CHARS]

            if entry.problem is not None:
                _blit_text(surface, fonts.body, num_str, (x_num, y), COLOR_DESTRUCTIVE)
                prob_str = f"{display_name}" + "  " + "—" + "  " + f"{entry.problem}"
                _blit_text(surface, fonts.body, prob_str, (x_name, y), COLOR_DESTRUCTIVE)
            else:
                _blit_text(surface, fonts.body, num_str, (x_num, y), COLOR_TEXT)
                _blit_text(surface, fonts.body, display_name, (x_name, y), COLOR_TEXT)
                detail_str = picker_entry_detail(entry.cols, entry.rows, entry.cars)
                _blit_text(surface, fonts.body, detail_str, (x_detail, y), COLOR_TEXT_MUTED)
            y += row_gap

    y_hint = WINDOW_SIZE[1] - SPACING_3XL - fonts.body.get_height()
    _blit_text(surface, fonts.body, PICKER_HINT, (SPACING_3XL, y_hint), COLOR_TEXT_MUTED)


def draw_load_error(
    surface: pygame.Surface,
    fonts: Fonts,
    file_name: str,
    detail: str,
) -> None:
    """Render the level load error screen composed entirely of theme tokens."""
    surface.fill(COLOR_BOARD_BG)
    _blit_text(surface, fonts.heading, LOAD_ERROR_HEADING, (SPACING_3XL, SPACING_3XL), COLOR_DESTRUCTIVE)

    y = SPACING_3XL + fonts.heading.get_height() + SPACING_MD
    _blit_text(surface, fonts.body, file_name, (SPACING_3XL, y), COLOR_TEXT)

    y += fonts.body.get_height() + SPACING_LG
    max_width = WINDOW_SIZE[0] - 2 * SPACING_3XL
    lines = _wrap_text(fonts.body, detail, max_width)[:LOAD_ERROR_MAX_LINES]
    for line in lines:
        _blit_text(surface, fonts.body, line, (SPACING_3XL, y), COLOR_DESTRUCTIVE)
        y += fonts.body.get_height() + SPACING_SM

    y_hint = WINDOW_SIZE[1] - SPACING_3XL - fonts.body.get_height()
    _blit_text(surface, fonts.body, LOAD_ERROR_HINT, (SPACING_3XL, y_hint), COLOR_TEXT_MUTED)
