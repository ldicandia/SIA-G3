"""Move rules, rejection reasons, and win detection.

``apply_move`` is the sole path by which a ``GameState`` may change --
the ui layer must never construct or hand-edit one directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gridworld.engine.board import Board, Position
from gridworld.engine.state import Direction, GameState


class MoveRejection(Enum):
    """Why a requested move was refused."""

    OFF_GRID = "off_grid"
    OBSTACLE = "obstacle"
    OCCUPIED = "occupied"
    FOREIGN_FLAG = "foreign_flag"
    CAR_PARKED = "car_parked"
    NO_SUCH_CAR = "no_such_car"


@dataclass(frozen=True, slots=True)
class MoveResult:
    """The outcome of a single ``apply_move`` call.

    When ``accepted`` is ``False``, ``state`` is the caller's own input
    object returned by identity -- never ``None`` and never a copy. This
    keeps the hot path explicit and non-throwing, and makes "no mutation
    of the source state" a one-line identity assertion.
    """

    accepted: bool
    state: GameState
    rejection: MoveRejection | None
    target: Position | None = None


def apply_move(board: Board, state: GameState, car: int, direction: Direction) -> MoveResult:
    """Attempt to move ``car`` one cell in ``direction``.

    Rejections are checked in this fixed order: unknown car, car already
    parked, off grid, obstacle, occupied by another car, foreign flag.
    Landing on the car's own flag parks it atomically with the move.
    """
    if car not in board.car_numbers():
        return MoveResult(accepted=False, state=state, rejection=MoveRejection.NO_SUCH_CAR, target=None)

    if state.is_parked(car):
        return MoveResult(accepted=False, state=state, rejection=MoveRejection.CAR_PARKED, target=state.position_of(car))

    row, col = state.position_of(car)
    drow, dcol = direction.delta
    target = (row + drow, col + dcol)

    if not board.in_bounds(target):
        return MoveResult(accepted=False, state=state, rejection=MoveRejection.OFF_GRID, target=target)

    if board.is_obstacle(target):
        return MoveResult(accepted=False, state=state, rejection=MoveRejection.OBSTACLE, target=target)

    occupant = state.car_at(target)
    if occupant is not None:
        return MoveResult(accepted=False, state=state, rejection=MoveRejection.OCCUPIED, target=target)

    flag_owner = board.flag_owner(target)
    if flag_owner is not None and flag_owner != car:
        return MoveResult(accepted=False, state=state, rejection=MoveRejection.FOREIGN_FLAG, target=target)

    new_cars = tuple(
        target if index == car - 1 else pos for index, pos in enumerate(state.cars)
    )
    new_parked = state.parked
    if target == board.flag_for(car):
        new_parked = state.parked | {car}

    new_state = GameState(cars=new_cars, parked=new_parked)
    return MoveResult(accepted=True, state=new_state, rejection=None, target=target)


@dataclass(frozen=True, slots=True)
class LegalMove:
    """One legal move a car may make from a given state.

    ``target`` is the cell the car lands on if this move is applied, so a
    caller (the renderer, in this milestone) never has to re-derive it from
    ``direction.delta``.
    """

    car: int
    direction: Direction
    target: Position


def legal_moves_for(board: Board, state: GameState, car: int) -> tuple[LegalMove, ...]:
    """Return every legal move for a single ``car``, in ``Direction`` order.

    Directions are tried in declaration order UP, DOWN, LEFT, RIGHT. Every
    decision is delegated to ``apply_move`` -- this function restates none
    of the six ``MoveRejection`` conditions itself, so enumeration and
    application can never disagree about whether a cell is reachable.
    Returns an empty tuple, never ``None``, for a parked car and for a car
    number ``board`` does not define.
    """
    moves: list[LegalMove] = []
    for direction in Direction:
        result = apply_move(board, state, car, direction)
        if result.accepted:
            moves.append(LegalMove(car=car, direction=direction, target=result.state.position_of(car)))
    return tuple(moves)


def legal_moves(board: Board, state: GameState) -> tuple[LegalMove, ...]:
    """Return every legal move from ``state``, across every car on ``board``.

    Ordering is deterministic and total: ascending car number first, then
    ``Direction`` declaration order (UP, DOWN, LEFT, RIGHT) within each car.
    Built by concatenating ``legal_moves_for`` over ``board.car_numbers()``,
    so there is a single enumeration body rather than two that could drift.
    Returns an empty tuple, never ``None``, when every car is parked or
    boxed in.
    """
    moves: list[LegalMove] = []
    for car in board.car_numbers():
        moves.extend(legal_moves_for(board, state, car))
    return tuple(moves)


def is_solved(board: Board, state: GameState) -> bool:
    """Return whether every car is parked on its own numbered flag."""
    for car in board.car_numbers():
        if car not in state.parked:
            return False
        if state.position_of(car) != board.flag_for(car):
            return False
    return True


def _reachable_cells(board: Board, state: GameState, car: int) -> frozenset[Position]:
    """Flood fill every cell ``car`` could eventually stand on, permanent blockers only.

    A breadth-first walk over grid *cells* (never game states) starting at
    ``state.position_of(car)``, using a plain list as a queue and a set of
    positions as the visited set. A neighbour cell is enterable when it is
    in bounds, is not an obstacle, carries either no flag or ``car``'s own
    flag, and is not occupied by a *parked* car.

    Unparked cars are deliberately treated as passable: they can still move
    out of the way, so blocking through them would manufacture a false
    alarm on every ordinary board where one car is briefly behind another
    -- exactly the failure the plan's prohibitions forbid.

    This is not the deferred TP search engine: it walks cells bounded by
    ``board.rows * board.cols``, not game states, and it exposes no
    frontier, cost, heuristic or metric -- only the set of cells reached.
    """
    start = state.position_of(car)
    visited: set[Position] = {start}
    queue: list[Position] = [start]

    while queue:
        row, col = queue.pop(0)
        for direction in Direction:
            drow, dcol = direction.delta
            neighbor = (row + drow, col + dcol)
            if neighbor in visited:
                continue
            if not board.in_bounds(neighbor):
                continue
            if board.is_obstacle(neighbor):
                continue
            flag_owner = board.flag_owner(neighbor)
            if flag_owner is not None and flag_owner != car:
                continue
            occupant = state.car_at(neighbor)
            if occupant is not None and state.is_parked(occupant):
                continue
            visited.add(neighbor)
            queue.append(neighbor)

    return frozenset(visited)


def stranded_cars(board: Board, state: GameState) -> tuple[int, ...]:
    """Return, in ascending order, every unparked car whose own flag is unreachable.

    Parked cars are skipped -- they have already reached their flag by
    definition. Returns an empty tuple when nothing is stranded.
    """
    stranded: list[int] = []
    for car in board.car_numbers():
        if state.is_parked(car):
            continue
        if board.flag_for(car) not in _reachable_cells(board, state, car):
            stranded.append(car)
    return tuple(stranded)


def is_unwinnable(board: Board, state: GameState) -> bool:
    """Return whether ``state`` can no longer be won.

    ``True`` when either branch holds: (A) ``stranded_cars`` is non-empty,
    or (B) the state is not solved and ``legal_moves`` is empty.

    **Soundness.** Every blocker branch A relies on is permanent within a
    single forward line of play: the grid boundary and obstacles never
    change, other cars' flags are refused unconditionally by ``apply_move``
    regardless of parked status, and ``apply_move`` itself never un-parks a
    car. ``MoveHistory.undo()`` *can* un-park a car and restore a previously
    lost route -- ``is_unwinnable`` stays sound there only because it is a
    pure function of the current ``(board, state)`` and is recomputed fresh
    on every transition (including undo), never cached across states.
    Branch B is trivially terminal: no legal move exists anywhere, by direct
    delegation to ``legal_moves`` from 03-01, so it restates none of the
    rejection ladder itself.

    **Incompleteness.** This is a sound but *incomplete* check: it never
    reports a winnable state as unwinnable, but it does not claim to catch
    every unwinnable state. A congestion deadlock that still admits at
    least one legal move -- for example two cars in a one-wide corridor
    that each need the cell the other occupies -- is not detected, because
    deciding that would require the state-space search this milestone
    explicitly excludes.
    """
    if stranded_cars(board, state):
        return True
    if not is_solved(board, state) and not legal_moves(board, state):
        return True
    return False
