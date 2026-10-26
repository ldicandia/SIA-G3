"""Kohonen: clases/2 - Modelo de Kohonen, pp. 13–15 y 24–37."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


TOPOLOGIES = ("hex", "rect")
DECAYS = ("inverse", "constant")
INITS = ("sample", "random")
ADJACENCY_TOL = 1e-9


@dataclass(frozen=True)
class SomGrid:
    rows: int
    cols: int
    topology: str
    positions: np.ndarray
    distances: np.ndarray

    @property
    def n_neurons(self) -> int:
        return self.rows * self.cols

    def neighbors(self, i: int) -> np.ndarray:
        near = np.flatnonzero(self.distances[i] <= 1.0 + ADJACENCY_TOL)
        return near[near != i]

    def adjacency_pairs(self) -> list[tuple[int, int]]:
        return [(a, int(b)) for a in range(self.n_neurons) for b in self.neighbors(a) if b > a]


def make_grid(rows: int, cols: int, topology: str) -> SomGrid:
    if type(rows) is not int or type(cols) is not int or rows < 1 or cols < 1 or rows * cols < 2:
        raise ValueError("grid needs positive integer dimensions and at least two neurons")
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}; allowed: {list(TOPOLOGIES)}")
    r, c = np.divmod(np.arange(rows * cols), cols)
    if topology == "hex":
        positions = np.column_stack([c + 0.5 * (r % 2), r * math.sqrt(3.0) / 2.0])
    else:
        positions = np.column_stack([c, r]).astype(float)
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    return SomGrid(rows, cols, topology, positions, distances)


@dataclass(frozen=True)
class SomConfig:
    rows: int
    cols: int
    topology: str = "hex"
    init: str = "sample"
    epochs: int = 200
    eta0: float = 0.5
    radius0: float = 3.0
    radius_decay: str = "inverse"


def decay(start: float, step: int, kind: str) -> float:
    """η(i)=η(0)/(1+i), clase p. 27; `constant` se usa para el radio (p. 31)."""
    if kind not in DECAYS:
        raise ValueError(f"unknown decay {kind!r}; allowed: {list(DECAYS)}")
    if not math.isfinite(start) or start <= 0 or step < 0:
        raise ValueError("decay needs a positive finite start and a nonnegative step")
    return float(start if kind == "constant" else start / (1 + step))


def neighborhood(distances, radius: float):
    """Vecindad binaria, incluyendo vecinos pegados con R=1 (clase, pp. 15 y 31)."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("neighborhood radius must be finite and positive")
    return np.where(np.asarray(distances) <= radius + ADJACENCY_TOL, 1.0, 0.0)


def init_weights(X: np.ndarray, n_neurons: int, method: str, rng: np.random.Generator) -> np.ndarray:
    if method not in INITS:
        raise ValueError(f"unknown init {method!r}; allowed: {list(INITS)}")
    if method == "random":
        return rng.uniform(X.min(axis=0), X.max(axis=0), size=(n_neurons, X.shape[1]))
    return np.array(X[rng.choice(len(X), n_neurons, replace=n_neurons > len(X))], dtype=float)


def squared_distances(X, W):
    return ((np.asarray(X)[:, None, :] - np.asarray(W)[None, :, :]) ** 2).sum(axis=2)


def best_matching_unit(W, x) -> int:
    return int(((W - x) ** 2).sum(axis=1).argmin())


def matching_units(X: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d2 = squared_distances(X, W)
    bmu = np.argmin(d2, axis=1)
    return bmu, np.sqrt(d2[np.arange(len(X)), bmu])


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
        ("init", config.init, INITS),
        ("radius_decay", config.radius_decay, DECAYS),
    ):
        if value not in allowed:
            raise ValueError(f"unknown {name} {value!r}; allowed: {list(allowed)}")
    if type(config.epochs) is not int or config.epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(config.eta0) or not 0 < config.eta0 < 1:
        raise ValueError("eta0 must satisfy 0 < eta0 < 1")
    if not math.isfinite(config.radius0) or config.radius0 < 1:
        raise ValueError("radius0 must be finite and at least 1")


def train_som(X: np.ndarray, config: SomConfig, rng: np.random.Generator) -> SomResult:
    _check_config(config)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite samples-by-features matrix")
    grid = make_grid(config.rows, config.cols, config.topology)
    W0 = init_weights(X, grid.n_neurons, config.init, rng)
    W = W0.copy()
    history = {key: [] for key in ("epoch", "eta", "radius", "mean_bmu_distance")}
    t = 0
    for epoch in range(config.epochs):
        for i in rng.permutation(len(X)):
            eta = decay(config.eta0, t, "inverse")
            radius = max(1.0, decay(config.radius0, t, config.radius_decay))
            x = X[int(i)]
            winner = best_matching_unit(W, x)
            h = neighborhood(grid.distances[winner], radius)
            W = W + (eta * h)[:, None] * (x - W)
            t += 1
        history["epoch"].append(epoch + 1)
        history["eta"].append(eta)
        history["radius"].append(radius)
        history["mean_bmu_distance"].append(float(matching_units(X, W)[1].mean()))
    return SomResult(config, grid, W, W0, {k: np.asarray(v) for k, v in history.items()})
