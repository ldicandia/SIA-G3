"""Board state treatments, HUD panel, illegal-move flash and win overlay.

Every colour, spacing value, font size and on-screen string is read from
``gridworld.ui.theme`` -- nothing here is written inline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import pygame

from gridworld.engine.board import Board
from gridworld.engine.rules import is_solved, legal_moves_for
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
    COLOR_SEARCH_FOCUS,
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
    LABEL_SEARCH,
    LEGAL_DESTINATION_ALPHA,
    LOAD_ERROR_HEADING,
    LOAD_ERROR_HINT,
    LOAD_ERROR_MAX_LINES,
    NO_CAR_SELECTED,
    PARKED_FILL_ALPHA,
    SEARCH_EXPLORED_ALPHA,
    SEARCH_PATH_ALPHA,
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
    WARNING_HEADING,
    WIN_BODY,
    WINDOW_SIZE,
    car_hue,
    cell_size,
    grid_origin,
    picker_entry_detail,
    unwinnable_body,
    unwinnable_hint,
    win_heading,
)


@dataclass(frozen=True)
class Fonts:
    """The four typography roles, built once at startup, never per frame."""

    body: pygame.font.Font
    label: pygame.font.Font
    heading: pygame.font.Font
    display: pygame.font.Font


@dataclass(frozen=True, slots=True)
class SearchHud:
    """Compact, renderer-only snapshot of the active optimal search."""

    status: str
    expanded_nodes: int
    frontier_nodes: int
    speed: str


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


def draw_legal_destinations(
    surface: pygame.Surface,
    board: Board,
    state: GameState,
    selected: int | None,
    cell: int | None = None,
    origin: tuple[int, int] | None = None,
) -> None:
    """Tint every cell ``selected`` may legally enter, in its own hue.

    Returns immediately when no car is selected. A parked or unknown
    selected car yields an empty enumeration from ``legal_moves_for``, so
    both cases self-handle with no extra branch. No text, no border, no
    state change.
    """
    if selected is None:
        return

    if cell is None or origin is None:
        cell = cell_size(board.cols, board.rows)
        origin_x, origin_y = grid_origin(board.cols, board.rows)
    else:
        origin_x, origin_y = origin

    hue = car_hue(selected)

    for move in legal_moves_for(board, state, selected):
        row, col = move.target
        rect = pygame.Rect(origin_x + col * cell, origin_y + row * cell, cell, cell)
        tint = pygame.Surface((cell, cell), pygame.SRCALPHA)
        tint.fill((*hue, LEGAL_DESTINATION_ALPHA))
        surface.blit(tint, rect.topleft)


def draw_board(
    surface: pygame.Surface,
    board: Board,
    state: GameState,
    selected: int | None,
    sprites: SpriteSet,
    explored_cells: Mapping[int, frozenset[tuple[int, int]]] | None = None,
    solution_paths: Mapping[int, tuple[tuple[int, int], ...]] | None = None,
    focus_cell: tuple[int, tuple[int, int]] | None = None,
) -> None:
    """Render every cell of ``board`` exactly once, in State Treatment order.

    Empty and obstacle cells first, then the selected car's legal
    destinations, then unclaimed flags, then cars (idle, parked, selected).
    The legal-destination tint sits beneath the flag and car sprites so it
    never covers a numeral. Drawing is clipped to the board area so a grid
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

    draw_legal_destinations(surface, board, state, selected, cell=cell, origin=(origin_x, origin_y))
    draw_search_trace(
        surface,
        board,
        explored_cells or {},
        solution_paths or {},
        focus_cell,
        cell,
        (origin_x, origin_y),
    )

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


def _grid_cells(rect: pygame.Rect, count: int) -> list[pygame.Rect]:
    """Split ``rect`` into ``count`` compact sub-rects, one per occupant.

    Several cars can land on the same board cell in the trace overlay; giving
    each occupant its own slice keeps every one of them visible instead of
    letting later draws fully cover -- or alpha-blend into an indistinct
    smear over -- earlier ones.
    """
    if count <= 1:
        return [rect]
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    cells: list[pygame.Rect] = []
    for index in range(count):
        row_idx, col_idx = divmod(index, cols)
        x0 = rect.x + round(col_idx * rect.width / cols)
        x1 = rect.x + round((col_idx + 1) * rect.width / cols)
        y0 = rect.y + round(row_idx * rect.height / rows)
        y1 = rect.y + round((row_idx + 1) * rect.height / rows)
        cells.append(pygame.Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0)))
    return cells


def draw_search_trace(
    surface: pygame.Surface,
    board: Board,
    explored_cells: Mapping[int, frozenset[tuple[int, int]]],
    solution_paths: Mapping[int, tuple[tuple[int, int], ...]],
    focus_cell: tuple[int, tuple[int, int]] | None,
    cell: int,
    origin: tuple[int, int],
) -> None:
    """Draw accumulated exploration dots and the final path beneath entities.

    Both are keyed by car number and drawn in that car's own hue, so each
    car's exploration and route stay visually distinct instead of blending
    into a single undifferentiated trace. When two or more cars share the
    same board cell, that cell is split via ``_grid_cells`` so every car's
    mark stays visible in its own slice rather than fully overlapping.
    """
    origin_x, origin_y = origin
    trace = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
    dot_radius = max(3, cell // 9)

    explored_by_cell: dict[tuple[int, int], list[int]] = {}
    for car, positions in explored_cells.items():
        for pos in positions:
            explored_by_cell.setdefault(pos, []).append(car)

    for (row, col), cars in explored_by_cell.items():
        rect = pygame.Rect(origin_x + col * cell, origin_y + row * cell, cell, cell)
        cars = sorted(cars)
        for car, sub in zip(cars, _grid_cells(rect, len(cars))):
            hue = car_hue(car)
            radius = dot_radius if len(cars) == 1 else max(2, min(sub.width, sub.height) // 2 - 1)
            pygame.draw.circle(trace, (*hue, SEARCH_EXPLORED_ALPHA), sub.center, radius)

    path_cell_cars: dict[tuple[int, int], list[int]] = {}
    for car, positions in solution_paths.items():
        seen: set[tuple[int, int]] = set()
        for pos in positions:
            if pos not in seen:
                path_cell_cars.setdefault(pos, []).append(car)
                seen.add(pos)

    for (row, col), cars in path_cell_cars.items():
        rect = pygame.Rect(origin_x + col * cell, origin_y + row * cell, cell, cell)
        cars = sorted(cars)
        for car, sub in zip(cars, _grid_cells(rect, len(cars))):
            hue = car_hue(car)
            pygame.draw.rect(trace, (*hue, SEARCH_PATH_ALPHA), sub)

    path_width = max(3, cell // 7)
    for car, positions in solution_paths.items():
        if not positions:
            continue
        hue = car_hue(car)
        centers = [
            (origin_x + col * cell + cell // 2, origin_y + row * cell + cell // 2)
            for row, col in positions
        ]
        if len(centers) > 1:
            pygame.draw.lines(
                trace,
                (*hue, 235),
                False,
                centers,
                width=path_width,
            )

    if focus_cell is not None:
        car, (row, col) = focus_cell
        hue = car_hue(car)
        center = (origin_x + col * cell + cell // 2, origin_y + row * cell + cell // 2)
        pygame.draw.circle(
            trace,
            (*hue, 245),
            center,
            dot_radius + max(2, cell // 18),
            width=max(2, cell // 20),
        )

    surface.blit(trace, (0, 0))


def draw_unwinnable_warning(
    surface: pygame.Surface,
    fonts: Fonts,
    origin: tuple[int, int],
    stranded: tuple[int, ...],
    moves: int,
) -> None:
    """Fill the reserved HUD warning block: heading, body, hint.

    Three lines stacked from ``origin``, separated by ``SPACING_XS``: the
    heading in the Label font in ``COLOR_DESTRUCTIVE``, the body from
    ``unwinnable_body`` in the Body font in ``COLOR_DESTRUCTIVE``, the hint
    from ``unwinnable_hint`` in the Body font in ``COLOR_TEXT_MUTED``.

    Height budget: the three lines plus their two gaps must fit within
    ``HUD_WARNING_BLOCK_HEIGHT``. If the bundled font's measured metrics do
    not fit, the heading is dropped and only body plus hint render -- the
    destructive colour on the body already carries the alarm.
    ``HUD_WARNING_BLOCK_HEIGHT`` itself is never grown: the whole point of
    the Phase 1 reservation is that this block's size was agreed before
    the content existed.
    """
    x, y = origin
    heading_label = fonts.label.render(WARNING_HEADING, True, COLOR_DESTRUCTIVE)
    body_label = fonts.body.render(unwinnable_body(stranded), True, COLOR_DESTRUCTIVE)
    hint_label = fonts.body.render(unwinnable_hint(moves), True, COLOR_TEXT_MUTED)

    full_height = (
        heading_label.get_height()
        + SPACING_XS
        + body_label.get_height()
        + SPACING_XS
        + hint_label.get_height()
    )

    if full_height <= HUD_WARNING_BLOCK_HEIGHT:
        surface.blit(heading_label, (x, y))
        y += heading_label.get_height() + SPACING_XS

    surface.blit(body_label, (x, y))
    y += body_label.get_height() + SPACING_XS
    surface.blit(hint_label, (x, y))


def draw_hud(
    surface: pygame.Surface,
    fonts: Fonts,
    moves: int,
    selected: int | None,
    unwinnable: bool = False,
    stranded: tuple[int, ...] = (),
    search: SearchHud | None = None,
) -> None:
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

    # Reserved for the unwinnable warning: blank unless `unwinnable` is set,
    # so adding it never reflows anything drawn above it.
    warning_rect = pygame.Rect(x, y, HUD_WIDTH - 2 * SPACING_LG, HUD_WARNING_BLOCK_HEIGHT)
    if search is not None:
        draw_search_hud(surface, fonts, warning_rect.topleft, search)
    elif unwinnable:
        draw_unwinnable_warning(surface, fonts, warning_rect.topleft, stranded, moves)


def draw_search_hud(
    surface: pygame.Surface,
    fonts: Fonts,
    origin: tuple[int, int],
    search: SearchHud,
) -> None:
    """Render search phase and metrics inside the reserved HUD block."""
    x, y = origin
    _blit_text(surface, fonts.label, LABEL_SEARCH, (x, y), COLOR_SEARCH_FOCUS)
    y += fonts.label.get_height() + SPACING_XS
    _blit_text(
        surface,
        fonts.body,
        f"{search.status} · {search.speed}",
        (x, y),
        COLOR_TEXT,
    )
    y += fonts.body.get_height() + SPACING_XS
    _blit_text(
        surface,
        fonts.body,
        f"Expanded {search.expanded_nodes} · Frontier {search.frontier_nodes}",
        (x, y),
        COLOR_TEXT_MUTED,
    )


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
    unwinnable: bool = False,
    stranded: tuple[int, ...] = (),
    explored_cells: Mapping[int, frozenset[tuple[int, int]]] | None = None,
    solution_paths: Mapping[int, tuple[tuple[int, int], ...]] | None = None,
    focus_cell: tuple[int, tuple[int, int]] | None = None,
    search: SearchHud | None = None,
    show_win_overlay: bool = True,
) -> None:
    """Compose a full frame: board, flash if active, HUD, win overlay if solved."""
    draw_board(
        surface,
        board,
        state,
        selected,
        sprites,
        explored_cells,
        solution_paths,
        focus_cell,
    )
    if flash_rect is not None:
        draw_illegal_flash(surface, flash_rect)
    draw_hud(surface, fonts, moves, selected, unwinnable, stranded, search)
    if show_win_overlay and is_solved(board, state):
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
