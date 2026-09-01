"""Contracts and tests for stop conditions and StopSet."""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from tp2.engine.config import ConfigError
from tp2.engine.events import GenerationContext
from tp2.engine.genome import Population
from tp2.engine.stop import StopSet, build_stop_set


def _make_ctx(generation: int, max_gen: int = 100, elapsed: float = 1.0, fitness: float = 0.5) -> GenerationContext:
    return GenerationContext(
        generation=generation,
        max_generations=max_gen,
        renders=100,
        elapsed=elapsed,
        fitness=np.array([fitness], dtype=np.float32),
    )


def test_empty_stop_set_raises_config_error():
    """A stop spec enabling none of the conditions raises ConfigError."""
    with pytest.raises(ConfigError, match="stop must enable at least one"):
        build_stop_set({"max_generations": False}, horizon=100)

    with pytest.raises(ConfigError, match="stop must enable at least one"):
        build_stop_set({}, horizon=100)


def test_max_generations_condition():
    """max_generations fires at generation == horizon when enabled, never when disabled."""
    stop_enabled = build_stop_set({"max_generations": True}, horizon=50)
    assert stop_enabled.check(_make_ctx(49)) == ""
    assert stop_enabled.check(_make_ctx(50)) == "max_generations"
    assert stop_enabled.check(_make_ctx(51)) == "max_generations"

    stop_disabled = build_stop_set({"max_generations": False, "wall_clock_seconds": 100.0}, horizon=50)
    assert stop_disabled.check(_make_ctx(50, elapsed=1.0)) == ""
    assert stop_disabled.check(_make_ctx(100, elapsed=1.0)) == ""


def test_wall_clock_condition():
    """wall_clock fires at elapsed >= wall_clock_seconds (inclusive), not before."""
    stop = build_stop_set({"max_generations": False, "wall_clock_seconds": 5.0}, horizon=100)
    assert stop.check(_make_ctx(10, elapsed=4.99)) == ""
    assert stop.check(_make_ctx(11, elapsed=5.0)) == "wall_clock"
    assert stop.check(_make_ctx(12, elapsed=5.5)) == "wall_clock"


def test_wall_clock_validation():
    """wall_clock_seconds must be a positive float."""
    with pytest.raises(ConfigError, match="wall_clock_seconds"):
        build_stop_set({"wall_clock_seconds": 0.0}, horizon=100)
    with pytest.raises(ConfigError, match="wall_clock_seconds"):
        build_stop_set({"wall_clock_seconds": -2.5}, horizon=100)


def test_min_fitness_condition():
    """min_fitness fires when max fitness >= min_fitness (inclusive), not before."""
    stop = build_stop_set({"max_generations": False, "min_fitness": 0.95}, horizon=100)
    ctx_low = GenerationContext(10, 100, 100, 1.0, np.array([0.90, 0.949], dtype=np.float32))
    ctx_hit = GenerationContext(11, 100, 110, 1.1, np.array([0.90, 0.950], dtype=np.float32))
    assert stop.check(ctx_low) == ""
    assert stop.check(ctx_hit) == "min_fitness"


def test_min_fitness_validation():
    """min_fitness must be in (0, 1]."""
    with pytest.raises(ConfigError, match="min_fitness"):
        build_stop_set({"min_fitness": 0.0}, horizon=100)
    with pytest.raises(ConfigError, match="min_fitness"):
        build_stop_set({"min_fitness": -0.1}, horizon=100)
    with pytest.raises(ConfigError, match="min_fitness"):
        build_stop_set({"min_fitness": 1.1}, horizon=100)


def test_content_stagnation_requires_full_window_and_detects_stall():
    """content_stagnation never fires before window is full, then fires when span < tolerance."""
    window = 4
    tol = 0.01
    stop = build_stop_set(
        {"max_generations": False, "content_stagnation": {"window": window, "tolerance": tol}},
        horizon=100,
    )

    # 3 flat generations (buffer length 1, 2, 3 < window): must NOT fire
    for g in range(3):
        ctx = _make_ctx(g, fitness=0.8)
        assert stop.check(ctx) == "", f"Fired early at generation {g}"

    # 4th flat generation (buffer length 4 == window): MUST fire
    ctx4 = _make_ctx(3, fitness=0.8)
    assert stop.check(ctx4) == "content_stagnation"

    # Moving fitness: span >= tolerance, does NOT fire
    stop_moving = build_stop_set(
        {"max_generations": False, "content_stagnation": {"window": window, "tolerance": tol}},
        horizon=100,
    )
    fitnesses = [0.80, 0.81, 0.83, 0.85]
    for g, f in enumerate(fitnesses):
        res = stop_moving.check(_make_ctx(g, fitness=f))
        assert res == ""


def test_structure_stagnation_streak_reset():
    """structure_stagnation requires streak of window consecutive qualifying generations."""
    window = 3
    stop = build_stop_set(
        {
            "max_generations": False,
            "structure_stagnation": {"window": window, "fraction": 0.8, "tolerance": 0.01},
        },
        horizon=100,
    )

    n = 10
    length = 22
    base_genes = np.full((n, length), 0.5, dtype=np.float32)
    base_fit = np.ones(n, dtype=np.float32)

    pop_same = Population(base_genes.copy(), base_fit.copy())
    pop_moved = Population(base_genes + 0.1, base_fit.copy())

    # Gen 0: initializes cache, streak = 0
    ctx = _make_ctx(0)
    assert stop.check(ctx, pop_same) == ""

    # Gen 1: qualify (streak 1)
    assert stop.check(_make_ctx(1), pop_same) == ""
    # Gen 2: qualify (streak 2)
    assert stop.check(_make_ctx(2), pop_same) == ""
    # Gen 3: DISQUALIFY (streak resets to 0)
    assert stop.check(_make_ctx(3), pop_moved) == ""

    # Gen 4: qualify (streak 1)
    assert stop.check(_make_ctx(4), pop_moved) == ""
    # Gen 5: qualify (streak 2)
    assert stop.check(_make_ctx(5), pop_moved) == ""
    # Gen 6: qualify (streak 3 == window -> FIRES)
    assert stop.check(_make_ctx(6), pop_moved) == "structure_stagnation"


def test_fixed_priority_order_independent_of_json_order():
    """When multiple conditions fire, StopSet returns the highest-priority condition."""
    # Write content_stagnation BEFORE min_fitness in dict (insertion order)
    spec = {
        "max_generations": False,
        "content_stagnation": {"window": 1, "tolerance": 0.1},
        "min_fitness": 0.8,
        "wall_clock_seconds": 10.0,
    }
    stop = build_stop_set(spec, horizon=100)

    # Context that satisfies min_fitness (0.9 >= 0.8), content_stagnation (tol 0.1), and wall_clock (15 >= 10)
    ctx = _make_ctx(10, elapsed=15.0, fitness=0.9)

    # Priority order: wall_clock > min_fitness > content_stagnation
    assert stop.check(ctx) == "wall_clock"

    # Context without wall_clock fired: min_fitness > content_stagnation
    ctx_fast = _make_ctx(10, elapsed=1.0, fitness=0.9)
    assert stop.check(ctx_fast) == "min_fitness"
