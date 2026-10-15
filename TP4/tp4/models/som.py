"""Kohonen self-organizing map written by hand in numpy (no SOM or clustering library).

Geometry: flat neuron index = row * cols + col. Hex maps use "odd-r" offset rows
(x = col + 0.5 * (row % 2), y = row * sqrt(3) / 2) so all six neighbors sit at
distance 1; rect maps use x = col, y = row. Grid distance is the Euclidean
distance between those positions.

Training is online: T = epochs * n_samples iterations, one creator at a time,
W += eta_t * h(d(bmu, .), r_t) * (x - W).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..backend import to_numpy

TOPOLOGIES = ("hex", "rect")
NEIGHBORHOODS = ("gaussian", "step")
DECAYS = ("exponential", "linear", "inverse", "constant")
INITS = ("sample", "random")
ADJACENCY_TOL = 1e-9


# --------------------------------------------------------------------------- grid


@dataclass(frozen=True)
class SomGrid:
    rows: int
    cols: int
    topology: str
    positions: np.ndarray  # (n, 2) x, y of every neuron
    distances: np.ndarray  # (n, n) symmetric Euclidean grid distances

    @property
    def n_neurons(self) -> int:
        return self.rows * self.cols

    def neighbors(self, i: int) -> np.ndarray:
        """Adjacent neuron indices (grid distance <= 1 + ADJACENCY_TOL), ascending, excluding i."""
        near = np.flatnonzero(self.distances[i] <= 1.0 + ADJACENCY_TOL)
        return near[near != i]

    def adjacency_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for a in range(self.n_neurons):
            pairs.extend((a, int(b)) for b in self.neighbors(a) if b > a)
        return pairs


def make_grid(rows: int, cols: int, topology: str) -> SomGrid:
    if rows < 1 or cols < 1:
        raise ValueError(f"grid needs rows >= 1 and cols >= 1, got {rows}x{cols}")
    if rows * cols < 2:
        raise ValueError(f"grid needs at least 2 neurons, got {rows}x{cols}")
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}; allowed: {list(TOPOLOGIES)}")
    r, c = np.divmod(np.arange(rows * cols), cols)
    if topology == "hex":
        x = c + 0.5 * (r % 2)
        y = r * (math.sqrt(3.0) / 2.0)
    else:
        x = c.astype(float)
        y = r.astype(float)
    positions = np.column_stack([x, y]).astype(float)
    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt((diff**2).sum(axis=2))
    return SomGrid(rows=rows, cols=cols, topology=topology, positions=positions, distances=distances)


# --------------------------------------------------------------------------- config and schedules


@dataclass(frozen=True)
class SomConfig:
    rows: int
    cols: int
    topology: str = "hex"
    neighborhood: str = "gaussian"
    init: str = "sample"
    epochs: int = 200
    eta0: float = 0.5
    eta_min: float = 0.01
    eta_decay: str = "exponential"
    radius0: float = 3.0
    radius_min: float = 1.0
    radius_decay: str = "exponential"


def decay(start: float, end: float, step: int, total: int, kind: str) -> float:
    """Schedule value at `step` of `total`: `start` at step 0, `end` at step total-1.

    p = step / max(total - 1, 1). exponential: start * (end / start)^p; linear: start + (end - start) * p;
    inverse: start / (1 + p * (start / end - 1)) (the course's 1/i family rescaled to hit `end`);
    constant: start. Values are clamped between start and end.
    """
    if kind not in DECAYS:
        raise ValueError(f"unknown decay {kind!r}; allowed: {list(DECAYS)}")
    if kind in ("exponential", "inverse") and not (start > 0 and end > 0):
        raise ValueError(f"{kind} decay needs start > 0 and end > 0, got {start}, {end}")
    if kind == "constant" or total <= 1 or step <= 0:
        return float(start)
    if step >= total - 1:
        return float(end)
    p = step / (total - 1)
    if kind == "exponential":
        value = start * (end / start) ** p
    elif kind == "linear":
        value = start + (end - start) * p
    else:  # inverse
        value = start / (1.0 + p * (start / end - 1.0))
    return float(min(max(value, min(start, end)), max(start, end)))


def neighborhood(distances, radius: float, kind: str, xp=np):
    """Influence of the winner on neurons at grid `distances`; 1 at distance 0 for every radius > 0.

    gaussian: exp(-d^2 / (2 r^2)), not truncated. step: 1 for d <= r (+ ADJACENCY_TOL), else 0.
    """
    if kind not in NEIGHBORHOODS:
        raise ValueError(f"unknown neighborhood {kind!r}; allowed: {list(NEIGHBORHOODS)}")
    if not radius > 0:
        raise ValueError(f"neighborhood radius must be > 0, got {radius}")
    d = xp.asarray(distances)
    if kind == "step":  # the course rule: 1 inside the radius (boundary included), 0 outside
        return xp.where(d <= radius + ADJACENCY_TOL, 1.0, 0.0)
    return xp.exp(-(d**2) / (2.0 * radius**2))


# --------------------------------------------------------------------------- BMU and metrics


def init_weights(X: np.ndarray, n_neurons: int, method: str, rng: np.random.Generator) -> np.ndarray:
    if method not in INITS:
        raise ValueError(f"unknown init {method!r}; allowed: {list(INITS)}")
    n_samples = X.shape[0]
    if method == "random":
        return rng.uniform(X.min(axis=0), X.max(axis=0), size=(n_neurons, X.shape[1]))
    idx = rng.choice(n_samples, n_neurons, replace=n_neurons > n_samples)
    return np.array(X[idx], dtype=float)


def squared_distances(X, W, xp=np):
    """(n_samples, n_neurons) squared Euclidean distances as explicit sums of squared differences."""
    X = xp.asarray(X)
    W = xp.asarray(W)
    return ((X[:, None, :] - W[None, :, :]) ** 2).sum(axis=2)


def best_matching_unit(W, x) -> int:
    """Index of the neuron nearest to x; ties go to the lowest index (first argmin)."""
    d2 = ((W - x) ** 2).sum(axis=1)
    return int(d2.argmin())


def best_two_units(X: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bmu, second bmu, distance to the bmu) per sample, from a stable argsort of squared distances."""
    d2 = squared_distances(X, W)
    order = np.argsort(d2, axis=1, kind="stable")
    bmu = order[:, 0]
    second = order[:, 1]
    bmu_distance = np.sqrt(d2[np.arange(d2.shape[0]), bmu])
    return bmu, second, bmu_distance


def quantization_error(X: np.ndarray, W: np.ndarray) -> float:
    return float(best_two_units(X, W)[2].mean())


def topographic_error(X: np.ndarray, W: np.ndarray, grid: SomGrid) -> float:
    bmu, second, _ = best_two_units(X, W)
    return float(np.mean(grid.distances[bmu, second] > 1.0 + ADJACENCY_TOL))


def distance_preservation(X: np.ndarray, W: np.ndarray, grid: SomGrid) -> float:
    """Pearson r between input-space distances of all creator pairs and the grid distances of their BMUs.

    NaN when undefined (fewer than 3 samples, or either distance vector is constant).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 3:
        return float("nan")
    bmu = best_two_units(X, W)[0]
    i, j = np.triu_indices(n, k=1)
    input_d = np.sqrt(((X[i] - X[j]) ** 2).sum(axis=1))
    grid_d = grid.distances[bmu[i], bmu[j]]
    if input_d.std() == 0.0 or grid_d.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(input_d, grid_d)[0, 1])


# --------------------------------------------------------------------------- training


@dataclass(frozen=True)
class SomResult:
    config: SomConfig
    grid: SomGrid
    weights: np.ndarray
    initial_weights: np.ndarray
    history: dict[str, np.ndarray]


def _check_config(config: SomConfig) -> None:
    for name, value, allowed in (
        ("topology", config.topology, TOPOLOGIES),
        ("neighborhood", config.neighborhood, NEIGHBORHOODS),
        ("init", config.init, INITS),
        ("eta_decay", config.eta_decay, DECAYS),
        ("radius_decay", config.radius_decay, DECAYS),
    ):
        if value not in allowed:
            raise ValueError(f"unknown {name} {value!r}; allowed: {list(allowed)}")


def train_som(X: np.ndarray, config: SomConfig, rng: np.random.Generator, xp=np) -> SomResult:
    _check_config(config)
    X = np.asarray(X, dtype=float)
    grid = make_grid(config.rows, config.cols, config.topology)
    n_samples = X.shape[0]
    W0 = init_weights(X, grid.n_neurons, config.init, rng)  # host RNG first: a fixed seed fixes everything
    W = xp.asarray(W0.copy())
    Xd = xp.asarray(X)
    dist = xp.asarray(grid.distances)
    total = config.epochs * n_samples

    hist: dict[str, list[float]] = {
        "epoch": [], "eta": [], "radius": [], "quantization_error": [], "topographic_error": [],
    }
    t = 0
    eta_t = config.eta0
    r_t = config.radius0
    for epoch in range(config.epochs):
        for i in rng.permutation(n_samples):
            eta_t = decay(config.eta0, config.eta_min, t, total, config.eta_decay)
            r_t = max(decay(config.radius0, config.radius_min, t, total, config.radius_decay), config.radius_min)
            x = Xd[int(i)]
            b = best_matching_unit(W, x)
            h = neighborhood(dist[b], r_t, config.neighborhood, xp)
            W = W + (eta_t * h)[:, None] * (x - W)
            t += 1
        Wh = to_numpy(W)
        hist["epoch"].append(epoch + 1)
        hist["eta"].append(eta_t)
        hist["radius"].append(r_t)
        hist["quantization_error"].append(quantization_error(X, Wh))
        hist["topographic_error"].append(topographic_error(X, Wh, grid))
    return SomResult(
        config=config,
        grid=grid,
        weights=to_numpy(W),
        initial_weights=W0,
        history={k: np.asarray(v, dtype=float) for k, v in hist.items()},
    )
