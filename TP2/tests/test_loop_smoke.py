from __future__ import annotations

import numpy as np

from tp2.engine.config import build_run_config
from tp2.engine.fitness import Evaluator
from tp2.engine.loop import Run


def _config(generations: int = 3):
    return build_run_config({
        "population": 6, "children": 6, "recombination_probability": 0.8,
        "parents": {"method": "elite"}, "replacement": {"method": "elite"},
        "crossover": {"method": "one_point", "boundary": "triangle"},
        "mutation": {"method": "gene", "probability": 0.5},
        "survival": {"method": "additive"}, "stop": {"max_generations": generations},
    })


def test_loop_emits_every_generation_without_rerendering_survivors() -> None:
    target = np.full((16, 16, 3), 255, dtype=np.float32)
    run = Run(_config(), Evaluator(target, (16, 16)), 3, np.random.default_rng(42))
    events = list(run)
    assert [event.generation for event in events] == [0, 1, 2, 3]
    assert [event.renders for event in events] == [6, 12, 18, 24]
    assert [event.best_fitness for event in events] == sorted(event.best_fitness for event in events)
    assert run.result is not None


def test_same_seed_has_same_metrics() -> None:
    target = np.full((16, 16, 3), 255, dtype=np.float32)
    def values():
        return [(event.renders, event.best_fitness) for event in Run(_config(2), Evaluator(target, (16, 16)), 3, np.random.default_rng(5))]
    assert values() == values()
