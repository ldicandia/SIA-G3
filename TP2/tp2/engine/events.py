"""Immutable values exchanged between the engine and its consumers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GenerationContext:
    generation: int
    max_generations: int
    renders: int
    elapsed: float
    fitness: np.ndarray


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    generation: int
    renders: int
    elapsed: float
    best_fitness: float
    best_error: float
    mean_fitness: float
    worst_fitness: float
    diversity: float
    active_triangles: int
    best_genes: np.ndarray
    best_frame: np.ndarray
    stop_reason: str


@dataclass(frozen=True, slots=True)
class RunResult:
    best_genes: np.ndarray
    best_frame: np.ndarray
    best_fitness: float
    best_error: float
    generations: int
    renders: int
    elapsed: float
    stop_reason: str
