"""Headless iterable generation loop."""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .config import RunConfig
from .events import GenerationEvent, RunResult
from .fitness import Evaluator
from .genome import Population, active_count, random_population


class Run:
    def __init__(self, config: RunConfig, evaluator: Evaluator, triangles: int, rng: np.random.Generator) -> None:
        self.config, self.evaluator, self.triangles, self.rng = config, evaluator, triangles, rng
        self.result: RunResult | None = None

    def _event(self, generation: int, population: Population, frame_cache: dict[bytes, np.ndarray], elapsed: float, reason: str) -> GenerationEvent:
        index = int(np.argmax(population.fitness))
        frame = frame_cache[population.genes[index].tobytes()]
        return GenerationEvent(generation, self.evaluator.renders, elapsed, float(population.fitness[index]),
                               1.0 - float(population.fitness[index]), float(population.fitness.mean()),
                               float(population.fitness.min()), float(population.genes.std(axis=0).mean()),
                               active_count(population.genes[index]), population.genes[index].copy(), frame, reason)

    def __iter__(self) -> Iterator[GenerationEvent]:
        started = time.perf_counter()
        genes = random_population(self.rng, self.config.population, self.triangles)
        fitness, frames = self.evaluator.evaluate_population(genes)
        population = Population(genes, fitness)
        frame_cache = {row.tobytes(): frame for row, frame in zip(genes, frames)}
        best = self._event(0, population, frame_cache, time.perf_counter() - started,
                           "max_generations" if self.config.max_generations == 0 else "")
        hall = best
        yield best
        for generation in range(1, self.config.max_generations + 1):
            parent_indices = self.config.parents(population.fitness, self.config.children, self.rng)
            offspring: list[np.ndarray] = []
            for start in range(0, self.config.children, 2):
                first = population.genes[parent_indices[start]]
                second = population.genes[parent_indices[(start + 1) % len(parent_indices)]]
                if self.rng.random() < self.config.recombination_probability:
                    child_1, child_2 = self.config.crossover(first, second, self.rng)
                else:
                    child_1, child_2 = first.copy(), second.copy()
                offspring.append(self.config.mutation(child_1, self.rng))
                if len(offspring) < self.config.children:
                    offspring.append(self.config.mutation(child_2, self.rng))
            child_genes = np.asarray(offspring, dtype=np.float32)
            child_fitness, child_frames = self.evaluator.evaluate_population(child_genes)
            frame_cache.update({row.tobytes(): frame for row, frame in zip(child_genes, child_frames)})
            children = Population(child_genes, child_fitness)
            population = self.config.survival(population, children, self.rng)
            event = self._event(generation, population, frame_cache, time.perf_counter() - started,
                                "max_generations" if generation == self.config.max_generations else "")
            if event.best_fitness > hall.best_fitness:
                hall = event
            yield event
        self.result = RunResult(hall.best_genes, hall.best_frame, hall.best_fitness, hall.best_error,
                                self.config.max_generations, self.evaluator.renders, time.perf_counter() - started,
                                "max_generations")
