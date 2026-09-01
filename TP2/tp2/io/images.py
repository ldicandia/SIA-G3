"""Target-image loading and PNG writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MAX_SOURCE_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS


def load_target(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    """Load an RGB target at an explicit canvas size as a float32 array."""
    source_path = Path(path)
    with Image.open(source_path) as source:
        width, height = source.size
        if width * height > MAX_SOURCE_PIXELS:
            raise ValueError(
                f"image {width}x{height} exceeds {MAX_SOURCE_PIXELS} source-pixel limit"
            )
        image = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8).astype(np.float32)


def save_png(frame: np.ndarray, path: str | Path) -> None:
    destination = Path(path)
    # Explicit encoder parameters (matching scripts/make_assets.py's
    # PNG_SAVE_KWARGS) so best.png does not depend on Pillow's per-release PNG
    # encoder defaults (01-VERIFICATION.md gap #1 / IO-08).
    Image.fromarray(np.asarray(frame, dtype=np.uint8), "RGB").save(
        destination, format="PNG", optimize=False, compress_level=6
    )
