"""Validated JSON configuration and operator wiring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .operators import crossover, mutation, selection, survival
from .operators.registry import CROSSOVER, MUTATION, SELECTION, SURVIVAL


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunConfig:
    population: int
    children: int
    max_generations: int
    recombination_probability: float
    parents: Callable
    replacement: Callable
    crossover: Callable
    mutation: Callable
    survival: Callable
    effective: dict[str, Any]


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
        population = _integer(data, "population", 1)
        children = _integer(data, "children", 1)
        stop = data["stop"]
        if not isinstance(stop, dict):
            raise ConfigError("stop must be an object")
        max_generations = _integer(stop, "max_generations", 0)
        probability = float(data["recombination_probability"])
        if not 0 <= probability <= 1:
            raise ConfigError(f"recombination_probability must be in [0, 1], got {probability!r}")
        parents_spec = data["parents"]
        replacement_spec = data["replacement"]
        crossover_spec = data["crossover"]
        mutation_spec = data["mutation"]
        survival_spec = data["survival"]
        if not all(isinstance(item, dict) for item in (parents_spec, replacement_spec, crossover_spec, mutation_spec, survival_spec)):
            raise ConfigError("operator settings must be objects")
        parent_selection = SELECTION.build(parents_spec)
        replacement = SELECTION.build(replacement_spec)
        return RunConfig(population, children, max_generations, probability, parent_selection, replacement,
                         CROSSOVER.build(crossover_spec), MUTATION.build(mutation_spec),
                         SURVIVAL.build({**survival_spec, "replacement": replacement}), data)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(str(exc)) from exc
