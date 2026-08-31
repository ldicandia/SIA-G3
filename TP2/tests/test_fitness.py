from __future__ import annotations

import numpy as np

from tp2.engine.fitness import FITNESS_FLOOR, Evaluator
from tp2.engine.genome import Population, random_population


def test_fitness_orientation_floor_and_exact_sse() -> None:
    target = np.zeros((8, 8, 3), dtype=np.float32)
    evaluator = Evaluator(target, (8, 8))
    one_pixel = np.zeros((8, 8, 3), dtype=np.uint8)
    one_pixel[0, 0, 0] = 1
    assert evaluator.sse(one_pixel) == 1.0
    assert evaluator.sse(np.zeros_like(one_pixel)) == 0.0
    assert Evaluator(np.zeros((8, 8, 3), dtype=np.float32), (8, 8)).evaluate(np.array([], dtype=np.float32))[0] == FITNESS_FLOOR


def test_render_counter_and_fitness_travel_with_population() -> None:
    evaluator = Evaluator(np.full((8, 8, 3), 255, dtype=np.float32), (8, 8))
    genes = random_population(np.random.default_rng(4), 3, 1)
    fitness, _ = evaluator.evaluate_population(genes)
    assert evaluator.renders == 3
    Population.concat(Population(genes[:1], fitness[:1]), Population(genes[1:], fitness[1:]))
    assert evaluator.renders == 3
