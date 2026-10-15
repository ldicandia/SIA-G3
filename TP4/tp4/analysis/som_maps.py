"""Map readings of a trained SOM: U-matrix, neighbor walls, prototypes, PC planes, regions.

numpy/pandas only: no file I/O, no RNG, no plotting. Everything here is a pure
function of the weight matrix (and the grid), so the same weights always give
the same numbers. The `som` experiment writes the results.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..models.som import SomGrid

# --------------------------------------------------------------------------- U-matrix


def u_matrix(weights: np.ndarray, grid: SomGrid) -> np.ndarray:
    """Mean Euclidean distance between each neuron's weights and its adjacent neurons' weights.

    This is the course definition of the U-matrix. NaN for a neuron without neighbors
    (impossible on a valid grid, guarded anyway).
    """
    W = np.asarray(weights, dtype=float)
    out = np.full(grid.n_neurons, np.nan)
    for i in range(grid.n_neurons):
        near = grid.neighbors(i)
        if near.size:
            out[i] = float(np.linalg.norm(W[near] - W[i], axis=1).mean())
    return out


def u_edges(weights: np.ndarray, grid: SomGrid) -> pd.DataFrame:
    """One row per adjacent neuron pair with the Euclidean distance between their weights.

    Sorted by distance descending, then neuron_a, then neuron_b (the strongest walls first).
    """
    W = np.asarray(weights, dtype=float)
    pairs = grid.adjacency_pairs()
    frame = pd.DataFrame(
        {
            "neuron_a": [a for a, _ in pairs],
            "neuron_b": [b for _, b in pairs],
            "distance": [float(np.linalg.norm(W[a] - W[b])) for a, b in pairs],
        },
        columns=["neuron_a", "neuron_b", "distance"],
    )
    frame = frame.sort_values(["distance", "neuron_a", "neuron_b"], ascending=[False, True, True], kind="stable")
    return frame.reset_index(drop=True)


def edge_feature_diffs(weights: np.ndarray, features: Sequence[str], a: int, b: int, n: int = 3) -> str:
    """The `n` features where neuron b differs most from neuron a, as "feature:+1.23;..." (w_b - w_a).

    Ordered by |difference| descending, ties by feature name.
    """
    diff = np.asarray(weights, dtype=float)[b] - np.asarray(weights, dtype=float)[a]
    order = sorted(range(len(features)), key=lambda i: (-abs(float(diff[i])), features[i]))
    return ";".join(f"{features[i]}:{float(diff[i]):+.2f}" for i in order[:n])
