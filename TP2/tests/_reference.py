"""Independent, deliberately slow alpha-compositing raster oracle."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def reference_render(genes: np.ndarray, size: tuple[int, int], background: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Textbook one-layer-per-triangle implementation; never share code with raster."""
    width, height = size
    canvas = Image.new("RGBA", size, (*background, 255))
    for row in np.asarray(genes).reshape(-1, 11):
        if row[10] < 0.5:
            continue
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        point_data = np.rint(row[:6].reshape(3, 2) * np.array([width, height])).astype(int)
        rgba = tuple(np.rint(row[6:10] * 255).astype(np.uint8).tolist())
        ImageDraw.Draw(layer).polygon([tuple(point) for point in point_data], fill=rgba)
        canvas = Image.alpha_composite(canvas, layer)
    return np.asarray(canvas.convert("RGB"), dtype=np.uint8)
