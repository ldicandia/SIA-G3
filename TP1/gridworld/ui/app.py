"""Game loop wiring the key map, move counter and flash timer to the renderer.

Every board change is rebound from ``apply_move``'s ``MoveResult`` -- this
layer never constructs or hand-edits a ``GameState`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys

import pygame

from gridworld.engine.board import Board, Position
from gridworld.engine.rules import apply_move, is_solved
from gridworld.engine.state import Direction, GameState
from gridworld.levelfile import Level, LevelEntry, LevelError, discover_levels, load_level
from gridworld.ui.render import Fonts, build_fonts, draw_frame, draw_load_error, draw_picker
from gridworld.ui.sprites import SpriteSet, build_sprites
from gridworld.ui.theme import FLASH_MS, PICKER_MAX_CHOICES, WINDOW_SIZE, WINDOW_TITLE, cell_size, grid_origin


class Screen(Enum):
    """The active screen rendered by the game loop."""

    CHOICE = "choice"
    GAME = "game"
    ERROR = "error"


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

    screen: Screen = Screen.CHOICE
    entries: tuple[LevelEntry, ...] = ()
    error_file: str = ""
    error_detail: str = ""
    level: Level | None = None
    board: Board | None = None
    state: GameState | None = None
    selected: int | None = None
    moves: int = 0
    flash_cell: Position | None = None
    flash_until_ms: int = 0


def _flash(session: Session, cell: Position) -> None:
    session.flash_cell = cell
    session.flash_until_ms = pygame.time.get_ticks() + FLASH_MS


def _handle_keydown(session: Session, key: int) -> None:
    if session.board is None or session.state is None or session.level is None:
        return

    if key == pygame.K_r:
        session.board = session.level.board
        session.state = session.level.state
        session.selected = None
        session.moves = 0
        session.flash_cell = None
        session.flash_until_ms = 0
        return

    if is_solved(session.board, session.state):
        # Only R and Esc respond once solved; movement and selection are
        # ignored so the win overlay cannot be dismissed by an arrow key.
        return

    if key in KEY_TO_CAR:
        car = KEY_TO_CAR[key]
        if car not in session.board.car_numbers():
            return
        if session.state.is_parked(car):
            # Selecting a parked car is refused: selection is unchanged and
            # the parked cell flashes destructive.
            _flash(session, session.state.position_of(car))
            return
        session.selected = car
        return

    if key in KEY_TO_DIRECTION:
        if session.selected is None:
            # No car selected: arrow keys do nothing and never reach apply_move.
            return
        direction = KEY_TO_DIRECTION[key]
        row, col = session.state.position_of(session.selected)
        drow, dcol = direction.delta
        target = (row + drow, col + dcol)
        result = apply_move(session.board, session.state, session.selected, direction)
        if result.accepted:
            session.state = result.state
            session.moves += 1
        else:
            _flash(session, target)


def _current_flash_rect(session: Session, cell: int) -> pygame.Rect | None:
    if session.board is None or session.flash_cell is None:
        return None
    if pygame.time.get_ticks() >= session.flash_until_ms:
        session.flash_cell = None
        return None
    origin_x, origin_y = grid_origin(session.board.cols, session.board.rows)
    row, col = session.flash_cell
    return pygame.Rect(origin_x + col * cell, origin_y + row * cell, cell, cell)


def _start_level(session: Session, level: Level) -> None:
    session.level = level
    session.board = level.board
    session.state = level.state
    session.selected = None
    session.moves = 0
    session.flash_cell = None
    session.flash_until_ms = 0
    session.screen = Screen.GAME


def run(level_path: str | Path | None = None) -> None:
    """Open the window and run the game loop until the player quits."""
    session = Session()

    if level_path is not None:
        try:
            level = load_level(level_path)
            _start_level(session, level)
        except LevelError as err:
            sys.stderr.write(f"{err}\n")
            sys.exit(1)
    else:
        session.entries = discover_levels()
        session.screen = Screen.CHOICE

    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    fonts: Fonts = build_fonts()
    sprites: SpriteSet | None = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if session.screen == Screen.CHOICE:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in KEY_TO_CAR:
                        choice_idx = KEY_TO_CAR[event.key] - 1
                        if 0 <= choice_idx < len(session.entries) and choice_idx < PICKER_MAX_CHOICES:
                            entry = session.entries[choice_idx]
                            if entry.problem is not None:
                                session.error_file = entry.path.name
                                session.error_detail = entry.problem
                                session.screen = Screen.ERROR
                            else:
                                try:
                                    level = load_level(entry.path)
                                    _start_level(session, level)
                                except LevelError as err:
                                    session.error_file = entry.path.name
                                    session.error_detail = err.detail
                                    session.screen = Screen.ERROR
                elif session.screen == Screen.ERROR:
                    if event.key == pygame.K_ESCAPE:
                        session.entries = discover_levels()
                        session.screen = Screen.CHOICE
                elif session.screen == Screen.GAME:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        _handle_keydown(session, event.key)

        if session.screen == Screen.CHOICE:
            draw_picker(screen, fonts, session.entries)
        elif session.screen == Screen.ERROR:
            draw_load_error(screen, fonts, session.error_file, session.error_detail)
        elif session.screen == Screen.GAME and session.board is not None and session.state is not None:
            current_cell = cell_size(session.board.cols, session.board.rows)
            if sprites is None or sprites.cell != current_cell or len(sprites.cars) != len(session.board.car_numbers()):
                sprites = build_sprites(session.board, current_cell)

            draw_frame(
                screen,
                fonts,
                session.board,
                session.state,
                session.selected,
                session.moves,
                sprites,
                flash_rect=_current_flash_rect(session, current_cell),
            )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
