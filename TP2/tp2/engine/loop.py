"""Headless iterable generation loop."""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .config import RunConfig
from .diversity import diversity
from .events import GenerationContext, GenerationEvent, RunResult
from .fitness import Evaluator
from .genome import Population, active_count, bounds_for, random_population


class Run:
    def __init__(self, config: RunConfig, evaluator: Evaluator, triangles: int, rng: np.random.Generator) -> None:
        self.config, self.evaluator, self.triangles, self.rng = config, evaluator, triangles, rng
        self.bounds = bounds_for(triangles)
        self.result: RunResult | None = None

    def _event(self, generation: int, population: Population, frame_cache: dict[bytes, np.ndarray], elapsed: float, reason: str) -> GenerationEvent:
        index = int(np.argmax(population.fitness))
        frame = frame_cache[population.genes[index].tobytes()]
        return GenerationEvent(generation, self.evaluator.renders, elapsed, float(population.fitness[index]),
                               1.0 - float(population.fitness[index]), float(population.fitness.mean()),
                               float(population.fitness.min()), diversity(population.genes, self.bounds),
                               active_count(population.genes[index]), population.genes[index].copy(), frame, reason)

    def __iter__(self) -> Iterator[GenerationEvent]:
        started = time.perf_counter()
        genes = random_population(self.rng, self.config.population, self.triangles)
        fitness, frames = self.evaluator.evaluate_population(genes)
        population = Population(genes, fitness)
        frame_cache = {row.tobytes(): frame for row, frame in zip(genes, frames)}
        elapsed = time.perf_counter() - started
        ctx = GenerationContext(0, self.config.horizon, self.evaluator.renders, elapsed, population.fitness)
        reason = self.config.stop.check(ctx, population)
        best = self._event(0, population, frame_cache, elapsed, reason)
        hall = best
        yield best
        if reason:
            self.result = RunResult(hall.best_genes, hall.best_frame, hall.best_fitness, hall.best_error,
                                    0, self.evaluator.renders, elapsed, reason)
            return

        generation = 0
        while True:
            generation += 1
            # This generation's context, built once and threaded into every
            # parents/mutation/survival(replacement) call below. Without it,
            # non_uniform mutation and boltzmann selection have no way to
            # know the real generation number (REVIEW.md CR-01/CR-02): the
            # fitness/renders/elapsed snapshot here is from *before* this
            # generation's offspring are evaluated, matching the same
            # before-this-iteration snapshot convention `hillclimber.py`
            # uses when it builds a `ctx` ahead of `mutate(...)`.
            gen_ctx = GenerationContext(generation, self.config.horizon, self.evaluator.renders, elapsed, population.fitness)
            parent_indices = self.config.parents(population.fitness, self.config.children, self.rng, ctx=gen_ctx)
            offspring: list[np.ndarray] = []
            # Pairs are formed in the order selection returned them -- (0,1),
            # (2,3), ... -- with no re-sort. `paired` is the largest even
            # count <= len(parent_indices); a strict `<` against a uniform
            # draw means probability 0.0 never recombines and 1.0 always
            # does.
            paired = len(parent_indices) - (len(parent_indices) % 2)
            for start in range(0, paired, 2):
                first = population.genes[parent_indices[start]]
                second = population.genes[parent_indices[start + 1]]
                if self.rng.random() < self.config.recombination_probability:
                    child_1, child_2 = self.config.crossover(first, second, self.rng)
                else:
                    child_1, child_2 = first.copy(), second.copy()
                offspring.append(self.config.mutation(child_1, self.rng, ctx=gen_ctx))
                offspring.append(self.config.mutation(child_2, self.rng, ctx=gen_ctx))
            if len(parent_indices) % 2:
                # Odd child count: the trailing parent has no partner. Per
                # the cátedra's rule for a non-recombined pairing, it is
                # unconditionally copied rather than crossed with a
                # wrapped-around partner, and it still passes through
                # mutation.
                leftover = population.genes[parent_indices[-1]]
                offspring.append(self.config.mutation(leftover.copy(), self.rng, ctx=gen_ctx))
            child_genes = np.asarray(offspring, dtype=np.float32)
            child_fitness, child_frames = self.evaluator.evaluate_population(child_genes)
            frame_cache.update({row.tobytes(): frame for row, frame in zip(child_genes, child_frames)})
            children = Population(child_genes, child_fitness)
            population = self.config.survival(population, children, self.rng, ctx=gen_ctx)
            # Bound frame_cache to the genes actually alive in the new
            # population: every future lookup in _event is scoped to
            # population.genes, and any bytes key dropped by survival's
            # replacement is never looked up again. Without this prune the
            # cache accumulates every genome's frame ever rendered for the
            # whole run (unbounded memory growth over long horizons).
            frame_cache = {row.tobytes(): frame_cache[row.tobytes()] for row in population.genes}
            elapsed = time.perf_counter() - started
            ctx = GenerationContext(generation, self.config.horizon, self.evaluator.renders, elapsed, population.fitness)
            reason = self.config.stop.check(ctx, population)
            event = self._event(generation, population, frame_cache, elapsed, reason)
            if event.best_fitness > hall.best_fitness:
                hall = event
            yield event
            if reason:
                self.result = RunResult(hall.best_genes, hall.best_frame, hall.best_fitness, hall.best_error,
                                        generation, self.evaluator.renders, elapsed, reason)
                return
