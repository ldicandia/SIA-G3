"""Validated JSON configuration and operator wiring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .operators import crossover, mutation, selection, survival
from .operators.registry import CROSSOVER, MUTATION, SELECTION, SURVIVAL
from .stop import StopSet, build_stop_set


class ConfigError(ValueError):
    pass


CONFIG_KEYS = frozenset({
    "population", "children", "horizon", "recombination_probability",
    "parents", "replacement", "crossover", "mutation", "survival", "stop",
})

# The two config slots that hold a selection spec -- both resolve through the
# same SELECTION registry (SUR-04), so a blend built for one works unchanged
# in the other.
SELECTION_SLOTS = ("parents", "replacement")


def desugar_selection(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize a selection spec into the nested form the registry consumes."""
    return spec


@dataclass(frozen=True, slots=True)
class RunConfig:
    population: int
    children: int
    horizon: int
    recombination_probability: float
    parents: Callable
    replacement: Callable
    crossover: Callable
    mutation: Callable
    survival: Callable
    stop: StopSet
    effective: dict[str, Any]

    @property
    def max_generations(self) -> int:
        return self.horizon


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid config {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ConfigError("config root must be an object")
    return result


def _integer(data: dict[str, Any], key: str, minimum: int) -> int:
    value = data.get(key)
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{key} must be an integer >= {minimum}, got {value!r}")
    return value


def build_run_config(data: dict[str, Any]) -> RunConfig:
    try:
        unknown = set(data) - CONFIG_KEYS
        if unknown:
            raise ConfigError(f"unknown config key(s): {sorted(unknown)}")
        population = _integer(data, "population", 1)
        children = _integer(data, "children", 1)
        horizon = _integer(data, "horizon", 1)
        stop_spec = data.get("stop")
        if not isinstance(stop_spec, dict):
            raise ConfigError("stop must be an object")
        stop = build_stop_set(stop_spec, horizon)

        probability = float(data["recombination_probability"])
        if not 0 <= probability <= 1:
            raise ConfigError(f"recombination_probability must be in [0, 1], got {probability!r}")

        selection_specs = {slot: data[slot] for slot in SELECTION_SLOTS}
        crossover_spec = data["crossover"]
        mutation_spec = data["mutation"]
        survival_spec = data["survival"]
        if not all(isinstance(item, dict) for item in (*selection_specs.values(), crossover_spec, mutation_spec, survival_spec)):
            raise ConfigError("operator settings must be objects")

        # Desugar in exactly one function, applied to both slots, so the
        # archived effective config below always records what actually ran.
        desugared = {slot: desugar_selection(spec) for slot, spec in selection_specs.items()}
        parent_selection = SELECTION.build(desugared["parents"])
        replacement = SELECTION.build(desugared["replacement"])

        effective = {**data, **desugared}
        return RunConfig(
            population=population,
            children=children,
            horizon=horizon,
            recombination_probability=probability,
            parents=parent_selection,
            replacement=replacement,
            crossover=CROSSOVER.build(crossover_spec),
            mutation=MUTATION.build(mutation_spec),
            survival=SURVIVAL.build({**survival_spec, "replacement": replacement}),
            stop=stop,
            effective=effective,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(str(exc)) from exc
