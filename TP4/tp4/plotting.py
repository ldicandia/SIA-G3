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
import numpy as np  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle, RegularPolygon  # noqa: E402

__all__ = ["plt", "save_figure", "grid_shape", "HEX_RADIUS", "draw_grid_cells"]

# Pointy-top hexagon whose flat-to-flat width is 1, so adjacent centers (1 apart) touch.
HEX_RADIUS = 1 / math.sqrt(3)


def save_figure(fig, path: str | Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def grid_shape(n: int, ncols: int = 6) -> tuple[int, int]:
    """(nrows, ncols) for n subplots laid out with at most `ncols` per row."""
    if n < 1:
        return (1, 1)
    cols = min(ncols, n)
    return (math.ceil(n / cols), cols)


def draw_grid_cells(
    ax,
    positions,
    topology: str,
    values=None,
    *,
    cmap: str = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
    colors=None,
    edgecolor: str = "white",
    linewidth: float = 0.8,
):
    """Draw one cell per SOM neuron (hex or rect) at `positions`; row 0 ends up at the top.

    Faces come from `values` through `cmap`/`vmin`/`vmax`, or from an explicit `colors` list.
    Returns the PatchCollection so callers can attach a colorbar.
    """
    pos = np.asarray(positions, dtype=float)
    if topology == "hex":
        patches = [RegularPolygon((x, y), numVertices=6, radius=HEX_RADIUS, orientation=0) for x, y in pos]
    elif topology == "rect":
        patches = [Rectangle((x - 0.5, y - 0.5), 1.0, 1.0) for x, y in pos]
    else:
        raise ValueError(f"unknown topology {topology!r}; allowed: ['hex', 'rect']")
    if colors is not None:
        collection = PatchCollection(patches, facecolors=list(colors), edgecolor=edgecolor, linewidth=linewidth)
    else:
        collection = PatchCollection(patches, cmap=cmap, edgecolor=edgecolor, linewidth=linewidth)
        collection.set_array(np.asarray(values if values is not None else np.zeros(len(pos)), dtype=float))
        collection.set_clim(vmin, vmax)
    ax.add_collection(collection)
    margin = 0.8
    ax.set_xlim(pos[:, 0].min() - margin, pos[:, 0].max() + margin)
    ax.set_ylim(pos[:, 1].max() + margin, pos[:, 1].min() - margin)  # inverted: row 0 on top
    ax.set_aspect("equal")
    ax.axis("off")
    return collection
