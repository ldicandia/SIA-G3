"""Sprite factory: builds car and flag surfaces per hue at a given cell size.

Every sprite is drawn in code with ``pygame.draw`` primitives -- no image
file is ever read, so there is no asset-missing failure mode. Sprites are
rebuilt only when the cell size changes, never per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from gridworld.engine.board import Board
from gridworld.ui.theme import SPACING_SM, car_hue, numeral_color, numeral_size


@dataclass(frozen=True)
class SpriteSet:
    """One car surface and one flag surface per 1-based car number.

    ``cell`` records the cell size these surfaces were built at, so a caller
    can detect a cell-size change and trigger a rebuild.
    """

    cell: int
    cars: dict[int, pygame.Surface]
    flags: dict[int, pygame.Surface]


def _digits(car: int) -> int:
    return len(str(car))


def _numeral_font(cell: int, digits: int) -> pygame.font.Font:
    size = numeral_size(cell, digits)
    return pygame.font.Font(pygame.font.get_default_font(), size)


def _stamp_numeral(surface: pygame.Surface, car: int, cell: int, center: tuple[int, int]) -> None:
    hue = car_hue(car)
    digits = _digits(car)
    font = _numeral_font(cell, digits)
    label = font.render(str(car), True, numeral_color(hue))
    label_rect = label.get_rect(center=center)
    surface.blit(label, label_rect)


def car_surface(car: int, cell: int) -> pygame.Surface:
    """Draw a car sprite: rounded body, cabin band, two wheels, numeral."""
    surface = pygame.Surface((cell, cell), pygame.SRCALPHA)
    hue = car_hue(car)
    inset = SPACING_SM
    body = pygame.Rect(inset, inset, cell - 2 * inset, cell - 2 * inset)
    body_radius = max(2, body.width // 6)
    pygame.draw.rect(surface, hue, body, border_radius=body_radius)

    cabin_height = max(2, body.height // 3)
    cabin = pygame.Rect(
        body.left + body.width // 6,
        body.top + max(1, body.height // 8),
        body.width - 2 * (body.width // 6),
        cabin_height,
    )
    cabin_color = tuple(min(255, c + 40) for c in hue)
    pygame.draw.rect(surface, cabin_color, cabin, border_radius=max(1, body_radius // 2))

    wheel_w = max(2, body.width // 6)
    wheel_h = max(2, body.height // 8)
    wheel_color = (30, 30, 30)
    left_wheel = pygame.Rect(body.left, body.bottom - wheel_h, wheel_w, wheel_h)
    right_wheel = pygame.Rect(body.right - wheel_w, body.bottom - wheel_h, wheel_w, wheel_h)
    pygame.draw.rect(surface, wheel_color, left_wheel)
    pygame.draw.rect(surface, wheel_color, right_wheel)

    _stamp_numeral(surface, car, cell, body.center)
    return surface


def flag_surface(car: int, cell: int) -> pygame.Surface:
    """Draw a flag sprite: vertical pole plus a right-pointing pennant, numeral."""
    surface = pygame.Surface((cell, cell), pygame.SRCALPHA)
    hue = car_hue(car)
    inset = SPACING_SM
    usable = cell - 2 * inset

    pole_width = max(2, usable // 8)
    pole = pygame.Rect(inset, inset, pole_width, usable)
    pygame.draw.rect(surface, hue, pole)

    pennant_top = inset
    pennant_height = max(4, usable // 2)
    pennant_left = pole.right
    pennant_width = max(4, usable - pole_width)
    pennant_points = [
        (pennant_left, pennant_top),
        (pennant_left + pennant_width, pennant_top + pennant_height // 2),
        (pennant_left, pennant_top + pennant_height),
    ]
    pygame.draw.polygon(surface, hue, pennant_points)

    pennant_center = (
        pennant_left + pennant_width // 3,
        pennant_top + pennant_height // 2,
    )
    _stamp_numeral(surface, car, cell, pennant_center)
    return surface


def _arrow_right(size: int, color: tuple[int, int, int]) -> pygame.Surface:
    """Draw a single right-pointing arrow: a shaft plus a triangular head."""
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    mid = size / 2

    shaft_height = max(1, round(size * 0.14))
    shaft = pygame.Rect(
        round(size * 0.10),
        round(mid - shaft_height / 2),
        round(size * 0.52),
        shaft_height,
    )
    pygame.draw.rect(surface, color, shaft)

    head = [
        (round(size * 0.92), round(mid)),
        (round(size * 0.54), round(size * 0.20)),
        (round(size * 0.54), round(size * 0.80)),
    ]
    pygame.draw.polygon(surface, color, head)
    return surface


_ARROW_ROTATION = {"right": 0, "up": 90, "left": 180, "down": 270}

ARROW_ORDER: tuple[str, str, str, str] = ("left", "up", "right", "down")
"""Reading order fixed by ``01-UI-SPEC.md``: ``← ↑ → ↓``."""


def arrow_surface(direction: str, size: int, color: tuple[int, int, int]) -> pygame.Surface:
    """Return one arrow glyph, drawn once rightward and rotated into place."""
    base = _arrow_right(size, color)
    angle = _ARROW_ROTATION[direction]
    return base if angle == 0 else pygame.transform.rotate(base, angle)


def arrow_row_surface(size: int, gap: int, color: tuple[int, int, int]) -> pygame.Surface:
    """Return the four legend arrows laid out left-to-right, ``gap`` apart."""
    arrows = [arrow_surface(direction, size, color) for direction in ARROW_ORDER]
    width = sum(arrow.get_width() for arrow in arrows) + gap * (len(arrows) - 1)
    height = max(arrow.get_height() for arrow in arrows)

    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    x = 0
    for arrow in arrows:
        surface.blit(arrow, (x, (height - arrow.get_height()) // 2))
        x += arrow.get_width() + gap
    return surface


def build_sprites(board: Board, cell: int) -> SpriteSet:
    """Build one car surface and one flag surface per car number on ``board``."""
    cars = {car: car_surface(car, cell) for car in board.car_numbers()}
    flags = {car: flag_surface(car, cell) for car in board.car_numbers()}
    return SpriteSet(cell=cell, cars=cars, flags=flags)
