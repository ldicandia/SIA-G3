"""Pillow rasterization of triangle chromosomes."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .genome import A, ACTIVE, ACTIVE_THRESHOLD, B, GENES_PER_TRIANGLE, G, R

BACKGROUND = (255, 255, 255)


def render(genes: np.ndarray, size: tuple[int, int], background: tuple[int, int, int] = BACKGROUND) -> np.ndarray:
    """Render active triangles in chromosome order as an RGB uint8 frame."""
    width, height = size
    triangles = np.asarray(genes).reshape(-1, GENES_PER_TRIANGLE)
    # RGB is load-bearing: RGBA here overwrites alpha rather than doing "over" blending.
    canvas = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(canvas, "RGBA")
    for triangle in triangles[triangles[:, ACTIVE] >= ACTIVE_THRESHOLD]:
        coords = np.rint(triangle[:6].reshape(3, 2) * np.array([width, height])).astype(int)
        color = tuple(np.rint(triangle[[R, G, B, A]] * 255).astype(np.uint8).tolist())
        draw.polygon([tuple(point) for point in coords], fill=color)
    return np.asarray(canvas, dtype=np.uint8)
