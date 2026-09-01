"""Fitness evaluation and honest render accounting."""

from __future__ import annotations

import math

import numpy as np

from .raster import BACKGROUND, render

FITNESS_FLOOR = 1e-12


class Evaluator:
    """Score candidates with positive higher-is-better fitness.

    The floor makes roulette selection well-defined even for maximally wrong
    frames, whose unfloored normalized-RMSE fitness would be zero.
    """

    def __init__(self, target: np.ndarray, size: tuple[int, int], background: tuple[int, int, int] = BACKGROUND) -> None:
        self._target = np.asarray(target, dtype=np.float32)
        self.size = size
        self.background = background
        expected = (size[1], size[0], 3)
        if self._target.shape != expected:
            raise ValueError(f"target shape must be {expected}, got {self._target.shape}")
        self.renders = 0

    @property
    def max_sse(self) -> float:
        return float(self._target.size) * 255.0 ** 2

    def sse(self, frame: np.ndarray) -> float:
        diff = (np.asarray(frame, dtype=np.float32) - self._target).ravel()
        # Accumulate in float64: float32 np.dot dispatches to BLAS sdot, which
        # previously returned 3196076288.0 (127 ULP low) instead of the exact
        # 3196108800.0 on the documented 128x128 full-contrast case
        # (01-VERIFICATION.md gap #2, code review CR-02). Do not revert this
        # to np.dot on the raw float32 diff.
        diff64 = diff.astype(np.float64)
        return float(np.dot(diff64, diff64))

    def evaluate(self, genes: np.ndarray) -> tuple[float, np.ndarray]:
        frame = render(genes, self.size, self.background)
        self.renders += 1
        error = math.sqrt(self.sse(frame) / self.max_sse)
        return max(1.0 - error, FITNESS_FLOOR), frame

    def evaluate_population(self, genes_matrix: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        outcomes = [self.evaluate(genes) for genes in genes_matrix]
        return (
            np.asarray([fitness for fitness, _ in outcomes], dtype=np.float32),
            [frame for _, frame in outcomes],
        )
