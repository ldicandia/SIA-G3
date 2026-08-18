"""Game loop wiring the key map, move counter and flash timer to the renderer.

Every board change is rebound from ``apply_move``'s ``MoveResult`` -- this
layer never constructs or hand-edits a ``GameState`` directly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys

import pygame

from gridworld.engine.board import Board, Position
from gridworld.engine.rules import LegalMove, apply_move, is_solved, is_unwinnable, stranded_cars
from gridworld.engine.state import Direction, GameState
from gridworld.history import MoveHistory
from gridworld.levelfile import Level, LevelEntry, LevelError, discover_levels, load_level
from gridworld.search.algorithms import AStarStepper
from gridworld.search.heuristics import heuristic_a
from gridworld.search.problem import Problem
from gridworld.ui.render import Fonts, SearchHud, build_fonts, draw_frame, draw_load_error, draw_picker
from gridworld.ui.sprites import SpriteSet, build_sprites
from gridworld.ui.theme import (
    FLASH_MS,
    PICKER_MAX_CHOICES,
    SEARCH_BATCH_SIZE,
    SEARCH_PATH_DELAYS_MS,
    SEARCH_PATH_PAUSE_MS,
    SEARCH_REVEAL_DELAYS_MS,
    SEARCH_TRAIL_REVEAL_MS,
    WINDOW_SIZE,
    WINDOW_TITLE,
    cell_size,
    grid_origin,
)


class Screen(Enum):
    """The active screen rendered by the game loop."""

    CHOICE = "choice"
    GAME = "game"
    ERROR = "error"


class SearchPhase(Enum):
    """Visible phases of an optimal-search animation."""

    EXPLORING = "exploring"
    DRAINING = "draining"
    PATH_PAUSE = "path_pause"
    TRAIL_REVEAL = "trail_reveal"
    REPLAYING = "replaying"
    COMPLETE = "complete"
    FAILED = "failed"


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

SEARCH_SPEED_LABELS = ("0.5x", "1x", "2x")


@dataclass
class SearchAnimation:
    """Non-blocking A* computation plus its two-phase visual playback."""

    stepper: AStarStepper
    base_history: MoveHistory
    phase: SearchPhase = SearchPhase.EXPLORING
    paused: bool = False
    speed_index: int = 0
    pending_cells: deque[tuple[int, Position]] = field(default_factory=deque)
    known_cells: dict[int, set[Position]] = field(default_factory=dict)
    explored_cells: dict[int, set[Position]] = field(default_factory=dict)
    focus_cell: tuple[int, Position] | None = None
    remaining_path: deque[LegalMove] = field(default_factory=deque)
    solution_paths: dict[int, list[Position]] = field(default_factory=dict)
    next_visual_ms: int = 0

    @property
    def expanded_nodes(self) -> int:
        return self.stepper.expanded_nodes

    @property
    def frontier_nodes(self) -> int:
        return self.stepper.frontier_nodes


@dataclass
class Session:
    """Session-local, presentation-only state -- never fed back into the engine.

    ``history`` is the session's source of truth for both the board
    position and the move count: there is no standalone current-state
    field and no standalone move-counter field, so the two cannot
    disagree. ``history.current`` is the board position; ``history.depth``
    is the move count.

    ``unwinnable`` and ``stranded`` are recomputed on state transition only
    -- never per frame -- so they always reflect ``history.current``
    without a flood fill running every tick.
    """

    screen: Screen = Screen.CHOICE
    entries: tuple[LevelEntry, ...] = ()
    error_file: str = ""
    error_detail: str = ""
    level: Level | None = None
    board: Board | None = None
    history: MoveHistory | None = None
    selected: int | None = None
    flash_cell: Position | None = None
    flash_until_ms: int = 0
    unwinnable: bool = False
    stranded: tuple[int, ...] = ()
    search: SearchAnimation | None = None


def _begin_search(session: Session) -> None:
    """Start an optimal A* search from the player's current state."""
    if session.board is None or session.history is None:
        return
    problem = Problem(session.board, session.history.current)
    session.search = SearchAnimation(
        stepper=AStarStepper(problem, heuristic_a),
        base_history=session.history,
        next_visual_ms=pygame.time.get_ticks(),
    )
    session.selected = None
    session.flash_cell = None
    session.flash_until_ms = 0


def _change_search_speed(search: SearchAnimation, delta: int) -> None:
    search.speed_index = max(0, min(len(SEARCH_SPEED_LABELS) - 1, search.speed_index + delta))
    search.next_visual_ms = pygame.time.get_ticks()


def build_solution_paths(
    board: Board, base_state: GameState, path: tuple[LegalMove, ...]
) -> dict[int, list[Position]]:
    """Replay ``path`` from ``base_state`` and return each car's full route.

    Used to trace the whole route at once -- before any car actually
    moves -- rather than growing it one step behind the animation.
    """
    solution_paths: dict[int, list[Position]] = {}
    state = base_state
    for action in path:
        source = state.position_of(action.car)
        outcome = apply_move(board, state, action.car, action.direction)
        car_path = solution_paths.setdefault(action.car, [source])
        if car_path[-1] != source:
            car_path.append(source)
        car_path.append(outcome.state.position_of(action.car))
        state = outcome.state
    return solution_paths


def _advance_search(session: Session, now_ms: int) -> None:
    """Advance computation and playback without blocking the event loop."""
    search = session.search
    if search is None or session.board is None or session.history is None or search.paused:
        return

    if search.phase is SearchPhase.EXPLORING:
        for expansion in search.stepper.advance(SEARCH_BATCH_SIZE):
            for car, position in enumerate(expansion.state.cars, start=1):
                car_known = search.known_cells.setdefault(car, set())
                if position not in car_known:
                    car_known.add(position)
                    search.pending_cells.append((car, position))

        result = search.stepper.result
        if result is not None:
            if result.success:
                search.remaining_path = deque(result.path)
                search.phase = SearchPhase.DRAINING
            else:
                search.phase = SearchPhase.FAILED

    if search.phase in (SearchPhase.EXPLORING, SearchPhase.DRAINING):
        if search.pending_cells and now_ms >= search.next_visual_ms:
            car, position = search.pending_cells.popleft()
            search.explored_cells.setdefault(car, set()).add(position)
            search.focus_cell = (car, position)
            search.next_visual_ms = now_ms + SEARCH_REVEAL_DELAYS_MS[search.speed_index]

        if search.phase is SearchPhase.DRAINING and not search.pending_cells:
            session.history = search.base_history
            search.focus_cell = None
            search.phase = SearchPhase.PATH_PAUSE
            search.next_visual_ms = now_ms + SEARCH_PATH_PAUSE_MS

    if search.phase is SearchPhase.PATH_PAUSE and now_ms >= search.next_visual_ms:
        search.solution_paths = build_solution_paths(
            session.board, search.base_history.current, tuple(search.remaining_path)
        )
        search.phase = SearchPhase.TRAIL_REVEAL
        search.next_visual_ms = now_ms + SEARCH_TRAIL_REVEAL_MS

    if search.phase is SearchPhase.TRAIL_REVEAL and now_ms >= search.next_visual_ms:
        search.phase = SearchPhase.REPLAYING
        search.next_visual_ms = now_ms

    if search.phase is SearchPhase.REPLAYING and now_ms >= search.next_visual_ms:
        if not search.remaining_path:
            search.phase = SearchPhase.COMPLETE
            session.selected = None
            _refresh_unwinnable(session)
            return

        action = search.remaining_path.popleft()
        outcome = apply_move(
            session.board,
            session.history.current,
            action.car,
            action.direction,
        )
        if not outcome.accepted:
            search.phase = SearchPhase.FAILED
            session.selected = None
            return

        session.history = session.history.push(outcome.state)
        session.selected = None if outcome.state.is_parked(action.car) else action.car
        search.next_visual_ms = now_ms + SEARCH_PATH_DELAYS_MS[search.speed_index]

        if not search.remaining_path:
            search.phase = SearchPhase.COMPLETE
            session.selected = None
            _refresh_unwinnable(session)


def _search_hud(search: SearchAnimation) -> SearchHud:
    if search.paused:
        status = "Paused"
    else:
        status = {
            SearchPhase.EXPLORING: "Exploring",
            SearchPhase.DRAINING: "Revealing explored cells",
            SearchPhase.PATH_PAUSE: "Optimal path found",
            SearchPhase.TRAIL_REVEAL: "Tracing optimal route",
            SearchPhase.REPLAYING: "Replaying optimal path",
            SearchPhase.COMPLETE: "Optimal solution complete",
            SearchPhase.FAILED: "No solution found",
        }[search.phase]
    return SearchHud(
        status=status,
        expanded_nodes=search.expanded_nodes,
        frontier_nodes=search.frontier_nodes,
        speed=SEARCH_SPEED_LABELS[search.speed_index],
    )


def _refresh_unwinnable(session: Session) -> None:
    """Recompute ``session.unwinnable`` and ``session.stranded`` from the current state.

    Called from exactly four state transitions -- level start, an accepted
    move, a successful undo, and reset -- and never from the frame loop, so
    the reachability walk runs once per transition rather than once per
    frame.
    """
    if session.board is None or session.history is None:
        return
    session.stranded = stranded_cars(session.board, session.history.current)
    session.unwinnable = is_unwinnable(session.board, session.history.current)


def _flash(session: Session, cell: Position) -> None:
    session.flash_cell = cell
    session.flash_until_ms = pygame.time.get_ticks() + FLASH_MS


def _handle_keydown(session: Session, key: int) -> None:
    if session.board is None or session.history is None or session.level is None:
        return

    if key == pygame.K_r:
        # The initial position already lives at the head of the history --
        # reset never reloads the level or touches session.level.
        session.history = session.history.reset()
        session.selected = None
        session.flash_cell = None
        session.flash_until_ms = 0
        session.search = None
        _refresh_unwinnable(session)
        return

    if key == pygame.K_s:
        if not is_solved(session.board, session.history.current):
            _begin_search(session)
        return

    if session.search is not None:
        if key == pygame.K_SPACE:
            session.search.paused = not session.search.paused
            session.search.next_visual_ms = pygame.time.get_ticks()
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            _change_search_speed(session.search, -1)
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            _change_search_speed(session.search, 1)
        return

    if is_solved(session.board, session.history.current):
        # Only R and Esc respond once solved; movement, selection and undo
        # are ignored so the win overlay cannot be dismissed by any of them.
        return

    if key == pygame.K_u:
        undone = session.history.undo()
        if undone is session.history:
            # Nothing to undo: no-op, no flash. The selection is
            # deliberately left untouched.
            return
        session.history = undone
        session.flash_cell = None
        session.flash_until_ms = 0
        _refresh_unwinnable(session)
        return

    if key in KEY_TO_CAR:
        car = KEY_TO_CAR[key]
        if car not in session.board.car_numbers():
            return
        if session.history.current.is_parked(car):
            # Selecting a parked car is refused: selection is unchanged and
            # the parked cell flashes destructive.
            _flash(session, session.history.current.position_of(car))
            return
        session.selected = car
        return

    if key in KEY_TO_DIRECTION:
        if session.selected is None:
            # No car selected: arrow keys do nothing and never reach apply_move.
            return
        direction = KEY_TO_DIRECTION[key]
        result = apply_move(session.board, session.history.current, session.selected, direction)
        if result.accepted:
            session.history = session.history.push(result.state)
            if result.state.is_parked(session.selected):
                session.selected = None
            _refresh_unwinnable(session)
        elif result.target is not None:
            _flash(session, result.target)


def _expire_and_get_flash_rect(session: Session, cell: int) -> pygame.Rect | None:
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
    session.history = MoveHistory.start(level.state)
    session.selected = None
    session.flash_cell = None
    session.flash_until_ms = 0
    session.screen = Screen.GAME
    session.search = None
    _refresh_unwinnable(session)


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
        elif session.screen == Screen.GAME and session.board is not None and session.history is not None:
            _advance_search(session, pygame.time.get_ticks())
            current_cell = cell_size(session.board.cols, session.board.rows)
            if sprites is None or sprites.cell != current_cell or len(sprites.cars) != len(session.board.car_numbers()):
                sprites = build_sprites(session.board, current_cell)

            search = session.search
            draw_frame(
                screen,
                fonts,
                session.board,
                session.history.current,
                session.selected,
                session.history.depth,
                sprites,
                flash_rect=_expire_and_get_flash_rect(session, current_cell),
                unwinnable=session.unwinnable,
                stranded=session.stranded,
                explored_cells=(
                    {car: frozenset(cells) for car, cells in search.explored_cells.items()}
                    if search is not None
                    else None
                ),
                solution_paths=(
                    {car: tuple(path) for car, path in search.solution_paths.items()}
                    if search is not None
                    else None
                ),
                focus_cell=search.focus_cell if search is not None else None,
                search=_search_hud(search) if search is not None else None,
                show_win_overlay=search is None,
            )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
