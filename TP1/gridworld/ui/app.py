"""Game loop wiring the key map, move counter and flash timer to the renderer.

Every board change is rebound from ``apply_move``'s ``MoveResult`` -- this
layer never constructs or hand-edits a ``GameState`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from gridworld.engine.board import Board, Position
from gridworld.engine.rules import apply_move, is_solved
from gridworld.engine.state import Direction, GameState
from gridworld.levels import built_in_level
from gridworld.ui.render import Fonts, build_fonts, draw_frame
from gridworld.ui.sprites import SpriteSet, build_sprites
from gridworld.ui.theme import FLASH_MS, WINDOW_SIZE, WINDOW_TITLE, cell_size, grid_origin

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
    flash_cell: Position | None = None
    flash_until_ms: int = 0


def _flash(session: Session, cell: Position) -> None:
    session.flash_cell = cell
    session.flash_until_ms = pygame.time.get_ticks() + FLASH_MS


def _handle_keydown(session: Session, key: int) -> None:
    if key == pygame.K_r:
        board, state = built_in_level()
        session.board = board
        session.state = state
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
    if session.flash_cell is None:
        return None
    if pygame.time.get_ticks() >= session.flash_until_ms:
        session.flash_cell = None
        return None
    origin_x, origin_y = grid_origin(session.board.cols, session.board.rows)
    row, col = session.flash_cell
    return pygame.Rect(origin_x + col * cell, origin_y + row * cell, cell, cell)


def run() -> None:
    """Open the window and run the game loop until the player quits."""
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    fonts: Fonts = build_fonts()

    board, state = built_in_level()
    session = Session(board=board, state=state)
    sprites: SpriteSet = build_sprites(board, cell_size(board.cols, board.rows))

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

        current_cell = cell_size(session.board.cols, session.board.rows)
        if current_cell != sprites.cell:
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
