"""Stop conditions and multi-predicate StopSet.

Implements all five cátedra stop criteria:
- max_generations (STP-01)
- wall_clock (STP-02)
- min_fitness / acceptable solution (STP-03)
- content_stagnation (STP-04)
- structure_stagnation (STP-05)

When multiple conditions fire simultaneously, StopSet resolves them via a fixed
priority order (STP-06).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from .diversity import unchanged_fraction

if TYPE_CHECKING:
    from .events import GenerationContext
    from .genome import Population

STOP_PRIORITY = (
    "max_generations",
    "wall_clock",
    "min_fitness",
    "content_stagnation",
    "structure_stagnation",
)


class StopSet:
    def __init__(
        self,
        horizon: int,
        max_generations_enabled: bool,
        wall_clock_seconds: float | None = None,
        min_fitness: float | None = None,
        content_stagnation: dict[str, Any] | None = None,
        structure_stagnation: dict[str, Any] | None = None,
    ) -> None:
        self.horizon = horizon
        self.max_generations_enabled = max_generations_enabled
        self.wall_clock_seconds = wall_clock_seconds
        self.min_fitness = min_fitness
        self.content_stagnation = content_stagnation
        self.structure_stagnation = structure_stagnation

        # Content stagnation ring buffer
        content_window = content_stagnation["window"] if content_stagnation else 50
        self._content_buffer: deque[float] = deque(maxlen=content_window)

        # Structure stagnation state
        self._structure_prev_genes: np.ndarray | None = None
        self._structure_prev_fitness: np.ndarray | None = None
        self._structure_streak: int = 0

    def check(self, ctx: GenerationContext, pop: Population | None = None) -> str:
        """Check all active stop conditions in fixed priority order."""
        best_f = float(np.max(ctx.fitness)) if ctx.fitness.size > 0 else 0.0
        self._content_buffer.append(best_f)

        # Evaluate structure stagnation state updates if pop is provided
        structure_fired = False
        if self.structure_stagnation is not None and pop is not None:
            if self._structure_prev_genes is None:
                self._structure_prev_genes = pop.genes.copy()
                self._structure_prev_fitness = pop.fitness.copy()
            else:
                tol = float(self.structure_stagnation["tolerance"])
                frac_target = float(self.structure_stagnation["fraction"])
                window = int(self.structure_stagnation["window"])

                frac = unchanged_fraction(
                    self._structure_prev_genes,
                    self._structure_prev_fitness,
                    pop.genes,
                    pop.fitness,
                    tol,
                )
                if frac >= frac_target:
                    self._structure_streak += 1
                else:
                    self._structure_streak = 0

                self._structure_prev_genes = pop.genes.copy()
                self._structure_prev_fitness = pop.fitness.copy()

                if self._structure_streak >= window:
                    structure_fired = True

        # Check in fixed priority order
        if self.max_generations_enabled and ctx.generation >= self.horizon:
            return "max_generations"

        if self.wall_clock_seconds is not None and ctx.elapsed >= self.wall_clock_seconds:
            return "wall_clock"

        if self.min_fitness is not None and (best_f >= self.min_fitness or np.isclose(best_f, self.min_fitness, atol=1e-6)):
            return "min_fitness"

        if self.content_stagnation is not None:
            win = int(self.content_stagnation["window"])
            tol = float(self.content_stagnation["tolerance"])
            if len(self._content_buffer) == win:
                if (max(self._content_buffer) - min(self._content_buffer)) < tol:
                    return "content_stagnation"

        if structure_fired:
            return "structure_stagnation"

        return ""


def build_stop_set(spec: dict[str, Any], horizon: int) -> StopSet:
    from tp2.engine.config import ConfigError

    if not isinstance(spec, dict):
        raise ConfigError(f"stop must be an object, got {spec!r}")
    if isinstance(horizon, bool) or type(horizon) is not int or horizon < 1:
        raise ConfigError(f"horizon must be an integer >= 1, got {horizon!r}")

    # Validate max_generations boolean
    max_gen_raw = spec.get("max_generations", False)
    if not isinstance(max_gen_raw, bool):
        raise ConfigError(f"stop.max_generations must be a boolean, got {max_gen_raw!r}")
    max_generations_enabled = max_gen_raw

    # Validate wall_clock_seconds
    wall_clock_seconds: float | None = None
    if "wall_clock_seconds" in spec:
        w_val = spec["wall_clock_seconds"]
        if isinstance(w_val, bool) or not isinstance(w_val, (int, float)) or w_val <= 0:
            raise ConfigError(f"stop.wall_clock_seconds must be a positive number, got {w_val!r}")
        wall_clock_seconds = float(w_val)

    # Validate min_fitness
    min_fitness: float | None = None
    if "min_fitness" in spec:
        f_val = spec["min_fitness"]
        if isinstance(f_val, bool) or not isinstance(f_val, (int, float)) or not (0.0 < f_val <= 1.0):
            raise ConfigError(f"stop.min_fitness must be in (0, 1], got {f_val!r}")
        min_fitness = float(f_val)

    # Validate content_stagnation
    content_stagnation: dict[str, Any] | None = None
    if "content_stagnation" in spec:
        c_spec = spec["content_stagnation"]
        if not isinstance(c_spec, dict):
            raise ConfigError(f"stop.content_stagnation must be an object, got {c_spec!r}")
        w = c_spec.get("window")
        t = c_spec.get("tolerance")
        if isinstance(w, bool) or type(w) is not int or w < 1:
            raise ConfigError(f"stop.content_stagnation.window must be an integer >= 1, got {w!r}")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or t <= 0:
            raise ConfigError(f"stop.content_stagnation.tolerance must be a positive number, got {t!r}")
        content_stagnation = {"window": int(w), "tolerance": float(t)}

    # Validate structure_stagnation
    structure_stagnation: dict[str, Any] | None = None
    if "structure_stagnation" in spec:
        s_spec = spec["structure_stagnation"]
        if not isinstance(s_spec, dict):
            raise ConfigError(f"stop.structure_stagnation must be an object, got {s_spec!r}")
        w = s_spec.get("window")
        f = s_spec.get("fraction")
        t = s_spec.get("tolerance")
        if isinstance(w, bool) or type(w) is not int or w < 1:
            raise ConfigError(f"stop.structure_stagnation.window must be an integer >= 1, got {w!r}")
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not (0.0 < f <= 1.0):
            raise ConfigError(f"stop.structure_stagnation.fraction must be in (0, 1], got {f!r}")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or t <= 0:
            raise ConfigError(f"stop.structure_stagnation.tolerance must be a positive number, got {t!r}")
        structure_stagnation = {"window": int(w), "fraction": float(f), "tolerance": float(t)}

    # Check that at least one stop condition is active
    if not (
        max_generations_enabled
        or wall_clock_seconds is not None
        or min_fitness is not None
        or content_stagnation is not None
        or structure_stagnation is not None
    ):
        raise ConfigError("stop must enable at least one stop condition; a run configured to never stop is a configuration error")

    return StopSet(
        horizon=horizon,
        max_generations_enabled=max_generations_enabled,
        wall_clock_seconds=wall_clock_seconds,
        min_fitness=min_fitness,
        content_stagnation=content_stagnation,
        structure_stagnation=structure_stagnation,
    )
