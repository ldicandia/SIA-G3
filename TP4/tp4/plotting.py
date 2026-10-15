"""Headless plotting helpers: Agg backend and one consistent save path.

Every plot module imports pyplot through this module so the backend is
selected before pyplot is first imported.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plt", "save_figure", "grid_shape"]


def save_figure(fig, path: str | Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def grid_shape(n: int, ncols: int = 6) -> tuple[int, int]:
    """(nrows, ncols) for n subplots laid out with at most `ncols` per row."""
    if n < 1:
        return (1, 1)
    cols = min(ncols, n)
    return (math.ceil(n / cols), cols)
