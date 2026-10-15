"""Experiment registry: one explicit entry per `kind`."""

from __future__ import annotations

from ..errors import ConfigError
from ..runs import Experiment
from . import eda, pca, som, som_control, sweep

REGISTRY: dict[str, Experiment] = {
    "eda": eda.EXPERIMENT,
    "pca": pca.EXPERIMENT,
    "som": som.EXPERIMENT,
    "som_control": som_control.EXPERIMENT,
    "sweep": sweep.EXPERIMENT,
}


def get_experiment(kind: str) -> Experiment:
    try:
        return REGISTRY[kind]
    except KeyError:
        raise ConfigError(f"unknown kind {kind!r}; known kinds: {sorted(REGISTRY)}") from None
