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


def test_sse_accumulates_in_float64_and_pins_the_exact_full_contrast_128x128_value() -> None:
    # Pre-fix, sse() accumulated in float32 via np.dot, which BLAS-dispatches
    # to sdot: this exact 128x128 full-contrast case returned 3196076288.0
    # (127 ULP low) instead of the true value asserted below. Exact equality
    # (not pytest.approx) is deliberate: an approximate comparison would hide
    # the dtype regression this test exists to catch.
    target = np.zeros((128, 128, 3), dtype=np.float32)
    evaluator = Evaluator(target, (128, 128))
    frame = np.full((128, 128, 3), 255, dtype=np.uint8)
    assert evaluator.sse(frame) == 3196108800.0
