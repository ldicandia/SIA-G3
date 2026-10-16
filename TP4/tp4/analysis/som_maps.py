"""Map readings of a trained SOM: U-matrix, neighbor walls, prototypes, PC planes, regions.

numpy/pandas only: no file I/O, no RNG, no plotting. Everything here is a pure
function of the weight matrix (and the grid), so the same weights always give
the same numbers. The `som` experiment writes the results.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

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


# --------------------------------------------------------------------------- planes in original units


def prototypes_original_units(
    weights: np.ndarray, mean: np.ndarray, std: np.ndarray, features: Sequence[str]
) -> pd.DataFrame:
    """Neuron prototypes back in the variables' own units: mean + w * std (the inverse of the z-score)."""
    W = np.asarray(weights, dtype=float)
    return pd.DataFrame(np.asarray(mean, dtype=float) + W * np.asarray(std, dtype=float), columns=list(features))


_ART_PARTS = ("topic_football_pct", "topic_makeup_pct", "topic_dogs_pct")


def reconstruct_topic_art(frame: pd.DataFrame) -> pd.Series:
    """topic_art_pct rebuilt as 100 - football - makeup - dogs (the topic block sums to 100)."""
    missing = [c for c in _ART_PARTS if c not in frame.columns]
    if missing:
        raise ValueError(f"cannot reconstruct topic_art_pct: missing column(s) {missing}")
    out = 100.0 - frame[list(_ART_PARTS)].sum(axis=1, skipna=False)
    return out.rename("topic_art_pct_rec")


def pc_projection(weights: np.ndarray, components: np.ndarray, n: int = 2) -> np.ndarray:
    """Project each row of `weights` on the first `n` principal axes: W @ components[:n].T."""
    return np.asarray(weights, dtype=float) @ np.asarray(components, dtype=float)[:n].T


# --------------------------------------------------------------------------- regions

MIN_REGION_CREATORS = 3
REGION_K_RANGE = (2, 8)
_TIE_TOL = 1e-12


def region_hierarchy(weights: np.ndarray, grid: SomGrid, k_values: Iterable[int]) -> dict[int, np.ndarray]:
    """Contiguity-constrained Ward agglomeration of the neuron prototypes.

    Starts from one region per neuron and repeatedly merges the pair of *adjacent* regions with
    the smallest Ward cost |A||B| / (|A| + |B|) * ||mean_A - mean_B||^2 (prototypes unweighted by
    hits), ties going to the lexicographically smallest (min neuron of A, min neuron of B).
    Returns, for every requested k, labels 1..k per neuron numbered by each region's smallest
    neuron index, so every region is a connected set of adjacent neurons and the output is a pure
    function of the weights.
    """
    W = np.asarray(weights, dtype=float)
    n = grid.n_neurons
    ks = sorted({int(k) for k in k_values})
    if not ks:
        return {}
    if ks[0] < 1 or ks[-1] > n:
        raise ValueError(f"k must be between 1 and {n} neurons, got {ks[0] if ks[0] < 1 else ks[-1]}")
    wanted = set(ks)

    members: dict[int, list[int]] = {i: [i] for i in range(n)}  # region id = its smallest neuron
    centroid = {i: W[i].copy() for i in range(n)}
    adj: dict[int, set[int]] = {i: {int(j) for j in grid.neighbors(i)} for i in range(n)}

    def cost(a: int, b: int) -> float:
        sa, sb = len(members[a]), len(members[b])
        diff = centroid[a] - centroid[b]
        return sa * sb / (sa + sb) * float(diff @ diff)

    costs = {(a, b): cost(a, b) for a in range(n) for b in adj[a] if b > a}

    def snapshot() -> np.ndarray:
        labels = np.zeros(n, dtype=int)
        for rank, rid in enumerate(sorted(members), start=1):
            labels[members[rid]] = rank
        return labels

    out: dict[int, np.ndarray] = {}
    while True:
        if len(members) in wanted:
            out[len(members)] = snapshot()
        if len(members) <= ks[0]:
            break
        if not costs:
            raise ValueError("the grid is not connected: cannot merge down to the requested number of regions")
        a, b = min(costs, key=lambda pair: (costs[pair], pair[0], pair[1]))
        size_a, size_b = len(members[a]), len(members[b])
        centroid[a] = (size_a * centroid[a] + size_b * centroid[b]) / (size_a + size_b)
        members[a] = sorted(members[a] + members.pop(b))
        del centroid[b]
        for c in adj[b]:
            adj[c].discard(b)
        adj[a] |= adj.pop(b)
        adj[a] -= {a, b}
        for c in adj[a]:
            adj[c].add(a)
        costs = {pair: v for pair, v in costs.items() if a not in pair and b not in pair}
        for c in adj[a]:
            lo, hi = (a, c) if a < c else (c, a)
            costs[(lo, hi)] = cost(lo, hi)
    return out


def segment_regions(weights: np.ndarray, grid: SomGrid, k: int) -> np.ndarray:
    return region_hierarchy(weights, grid, [k])[int(k)]


def region_border_ratio(weights: np.ndarray, grid: SomGrid, labels: np.ndarray) -> float:
    """Mean weight distance across region borders divided by the mean inside regions (adjacent pairs).

    NaN when there are no border pairs, no inside pairs, or the inside mean is zero.
    """
    W = np.asarray(weights, dtype=float)
    lab = np.asarray(labels)
    across: list[float] = []
    inside: list[float] = []
    for a, b in grid.adjacency_pairs():
        (across if lab[a] != lab[b] else inside).append(float(np.linalg.norm(W[a] - W[b])))
    if not across or not inside:
        return float("nan")
    base = float(np.mean(inside))
    return float(np.mean(across)) / base if base > 0 else float("nan")


def region_scan(weights: np.ndarray, grid: SomGrid, bmu: np.ndarray, k_values: Iterable[int]) -> pd.DataFrame:
    """One row per k: border/inside U ratio, smallest creator count of any region, regions without creators."""
    ks = sorted({int(k) for k in k_values})
    columns = ["k", "u_border_ratio", "min_creators", "n_regions_without_creators"]
    if not ks:
        return pd.DataFrame(columns=columns)
    hierarchy = region_hierarchy(weights, grid, ks)
    bmu = np.asarray(bmu, dtype=int)
    rows = []
    for k in ks:
        labels = hierarchy[k]
        counts = np.bincount(labels[bmu], minlength=k + 1)[1:]
        rows.append(
            {
                "k": k,
                "u_border_ratio": region_border_ratio(weights, grid, labels),
                "min_creators": int(counts.min()),
                "n_regions_without_creators": int((counts == 0).sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def choose_n_regions(scan: pd.DataFrame, min_creators: int = MIN_REGION_CREATORS) -> int:
    """k with the highest u_border_ratio among rows whose regions all hold >= `min_creators` creators.

    Ties go to the smaller k; k = 2 when no row qualifies (or the scan is empty).
    """
    best_k, best_ratio = 2, -np.inf
    for row in scan.sort_values("k", kind="stable").itertuples(index=False):
        ratio = float(row.u_border_ratio)
        if int(row.min_creators) < min_creators or np.isnan(ratio):
            continue
        if ratio > best_ratio + _TIE_TOL:
            best_k, best_ratio = int(row.k), ratio
    return best_k


def correlation_ratio(values: np.ndarray, labels: np.ndarray) -> float:
    """eta^2 = SS_between / SS_total of `values` grouped by `labels` (0 when SS_total is 0)."""
    v = np.asarray(values, dtype=float)
    lab = np.asarray(labels)
    total = float(((v - v.mean()) ** 2).sum())
    if not total > 0:
        return 0.0
    between = 0.0
    for r in np.unique(lab):
        group = v[lab == r]
        between += group.size * (float(group.mean()) - float(v.mean())) ** 2
    return float(between / total)


def region_profiles(Z: pd.DataFrame, raw: pd.DataFrame, creator_regions: np.ndarray, k: int) -> pd.DataFrame:
    """Long table region x feature: mean z-score and mean raw value over the region's creators.

    Covers the selected features of `Z` plus topic_art_pct (z-scored from its own raw column,
    ddof=0). A region with no creators gets n_creators 0 and NaN means.
    """
    regions = np.asarray(creator_regions, dtype=int)
    art = raw["topic_art_pct"].to_numpy(dtype=float)
    art_std = float(art.std(ddof=0))
    art_z = (art - art.mean()) / art_std if art_std > 0 else np.zeros_like(art)
    features = list(Z.columns) + ["topic_art_pct"]
    z_values = {**{c: Z[c].to_numpy(dtype=float) for c in Z.columns}, "topic_art_pct": art_z}
    raw_values = {**{c: raw[c].to_numpy(dtype=float) for c in Z.columns}, "topic_art_pct": art}
    rows = []
    for r in range(1, int(k) + 1):
        mask = regions == r
        n = int(mask.sum())
        for f in features:
            rows.append(
                {
                    "region": r,
                    "feature": f,
                    "mean_z": float(z_values[f][mask].mean()) if n else float("nan"),
                    "mean_raw": float(raw_values[f][mask].mean()) if n else float("nan"),
                    "n_creators": n,
                }
            )
    return pd.DataFrame(rows, columns=["region", "feature", "mean_z", "mean_raw", "n_creators"])
