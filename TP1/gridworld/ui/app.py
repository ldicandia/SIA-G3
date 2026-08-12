"""Plain-rectangle pygame window for the Grid World walking skeleton.

Deliberately unstyled: plain `pygame.draw.rect` primitives and the
default font only. The sprite factory, Okabe-Ito palette, HUD styling,
illegal-move flash, and win overlay scrim from `01-UI-SPEC.md` are owned
by plan 01-03 and are out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from gridworld.engine.board import Board
from gridworld.engine.rules import apply_move, is_solved
from gridworld.engine.state import Direction, GameState
from gridworld.levels import built_in_level

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 640
BOARD_AREA_SIZE = 640
HUD_PANEL_WIDTH = 320
WINDOW_TITLE = "Grid World — TP1"

COLOR_EMPTY = (217, 217, 217)  # grid line color; empty cell fill is white
COLOR_WHITE = (255, 255, 255)
COLOR_OBSTACLE = (26, 26, 26)
COLOR_HUD_BG = (242, 242, 242)
COLOR_TEXT = (26, 26, 26)
COLOR_SELECTED_OUTLINE = (26, 26, 26)

CAR_COLORS = {
    1: (230, 159, 0),
    2: (86, 180, 233),
    3: (0, 158, 115),
}

KEY_TO_CAR = {
    pygame.K_1: 1,
    pygame.K_2: 2,
    pygame.K_3: 3,
    pygame.K_4: 4,
    pygame.K_5: 5,
    pygame.K_6: 6,
    pygame.K_7: 7,
    pygame.K_8: 8,
    pygame.K_9: 9,
}

KEY_TO_DIRECTION = {
    pygame.K_UP: Direction.UP,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
}


@dataclass
class Session:
    """Session-local, presentation-only state -- never fed back into the engine."""

    board: Board
    state: GameState
    selected: int | None = None
    moves: int = 0


def _cell_size(board: Board) -> int:
    return max(24, (BOARD_AREA_SIZE - 64) // max(board.cols, board.rows))


def _car_color(car: int) -> tuple[int, int, int]:
    return CAR_COLORS.get(((car - 1) % len(CAR_COLORS)) + 1, (140, 140, 140))


def _handle_keydown(session: Session, key: int) -> None:
    solved = is_solved(session.board, session.state)

    if key == pygame.K_r:
        board, state = built_in_level()
        session.board = board
        session.state = state
        session.selected = None
        session.moves = 0
        return

    if solved:
        return

    if key in KEY_TO_CAR:
        car = KEY_TO_CAR[key]
        if car in session.board.car_numbers() and not session.state.is_parked(car):
            session.selected = car
        return

    if key in KEY_TO_DIRECTION and session.selected is not None:
        direction = KEY_TO_DIRECTION[key]
        result = apply_move(session.board, session.state, session.selected, direction)
        session.state = result.state
        if result.accepted:
            session.moves += 1


def _draw_board(surface: pygame.Surface, session: Session, font: pygame.font.Font) -> None:
    board = session.board
    cell = _cell_size(board)
    grid_width = board.cols * cell
    grid_height = board.rows * cell
    origin_x = max(0, (BOARD_AREA_SIZE - grid_width) // 2)
    origin_y = max(0, (BOARD_AREA_SIZE - grid_height) // 2)

    surface.fill(COLOR_WHITE, pygame.Rect(0, 0, BOARD_AREA_SIZE, BOARD_AREA_SIZE))

    for row in range(board.rows):
        for col in range(board.cols):
            rect = pygame.Rect(origin_x + col * cell, origin_y + row * cell, cell, cell)
            if rect.right > BOARD_AREA_SIZE or rect.bottom > BOARD_AREA_SIZE:
                continue
            pos = (row, col)
            if board.is_obstacle(pos):
                pygame.draw.rect(surface, COLOR_OBSTACLE, rect)
            else:
                pygame.draw.rect(surface, COLOR_WHITE, rect)
            pygame.draw.rect(surface, COLOR_EMPTY, rect, width=1)

    for car in board.car_numbers():
        flag_pos = board.flag_for(car)
        if car in session.state.parked:
            continue
        row, col = flag_pos
        rect = pygame.Rect(origin_x + col * cell, origin_y + row * cell, cell, cell)
        pygame.draw.rect(surface, _car_color(car), rect.inflate(-8, -8))
        _blit_numeral(surface, font, str(car), rect.center)

    for car in board.car_numbers():
        row, col = session.state.position_of(car)
        rect = pygame.Rect(origin_x + col * cell, origin_y + row * cell, cell, cell)
        color = _car_color(car)
        if session.state.is_parked(car):
            pygame.draw.rect(surface, color, rect, width=3)
            pygame.draw.rect(surface, color, rect.inflate(-8, -8))
        else:
            pygame.draw.rect(surface, color, rect.inflate(-8, -8))
            if session.selected == car:
                pygame.draw.rect(surface, COLOR_SELECTED_OUTLINE, rect, width=3)
        _blit_numeral(surface, font, str(car), rect.center)


def _blit_numeral(
    surface: pygame.Surface, font: pygame.font.Font, text: str, center: tuple[int, int]
) -> None:
    label = font.render(text, True, COLOR_WHITE)
    label_rect = label.get_rect(center=center)
    surface.blit(label, label_rect)


def _draw_hud(
    surface: pygame.Surface,
    session: Session,
    label_font: pygame.font.Font,
    value_font: pygame.font.Font,
) -> None:
    panel_rect = pygame.Rect(BOARD_AREA_SIZE, 0, HUD_PANEL_WIDTH, WINDOW_HEIGHT)
    surface.fill(COLOR_HUD_BG, panel_rect)

    x = BOARD_AREA_SIZE + 24
    y = 24

    moves_label = label_font.render("Moves", True, COLOR_TEXT)
    surface.blit(moves_label, (x, y))
    y += 20
    moves_value = value_font.render(str(session.moves), True, COLOR_TEXT)
    surface.blit(moves_value, (x, y))
    y += 48

    car_label = label_font.render("Car", True, COLOR_TEXT)
    surface.blit(car_label, (x, y))
    y += 20
    car_text = str(session.selected) if session.selected is not None else "—"
    car_value = value_font.render(car_text, True, COLOR_TEXT)
    surface.blit(car_value, (x, y))
    y += 48

    controls_label = label_font.render("Controls", True, COLOR_TEXT)
    surface.blit(controls_label, (x, y))
    y += 24
    for line in ("Arrows  move", "1-9  select car", "R  reset", "Esc  quit"):
        line_surface = label_font.render(line, True, COLOR_TEXT)
        surface.blit(line_surface, (x, y))
        y += 20

    if is_solved(session.board, session.state):
        heading = f"Solved in {session.moves} move" + ("" if session.moves == 1 else "s")
        heading_surface = value_font.render(heading, True, COLOR_TEXT)
        surface.blit(heading_surface, (x, y + 20))


def run() -> None:
    """Open the window and run the game loop until the player quits."""
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    label_font = pygame.font.Font(pygame.font.get_default_font(), 13)
    value_font = pygame.font.Font(pygame.font.get_default_font(), 24)
    numeral_font = pygame.font.Font(pygame.font.get_default_font(), 16)

    board, state = built_in_level()
    session = Session(board=board, state=state)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    _handle_keydown(session, event.key)

        _draw_board(screen, session, numeral_font)
        _draw_hud(screen, session, label_font, value_font)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
