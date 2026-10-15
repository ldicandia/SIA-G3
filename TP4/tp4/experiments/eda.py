"""Experiment kind `eda`: exploratory analysis of creadores.csv."""

from __future__ import annotations

from typing import Any

from ..analysis.eda import describe_columns
from ..data import load_creators, numeric_columns
from ..errors import ConfigError
from ..runs import Experiment, RunContext, resolve_project_path, write_csv

PARAM_DEFAULTS: dict[str, Any] = {"outlier_iqr_k": 1.5}
EXTRA_KEYS = frozenset({"data"})


def validate(config: dict) -> None:
    data = config.get("data")
    if not isinstance(data, str) or not data:
        raise ConfigError(f"eda: data must be a non-empty string path, got {data!r}")
    k = config["params"]["outlier_iqr_k"]
    if not k > 0:
        raise ConfigError(f"eda: params.outlier_iqr_k must be > 0, got {k!r}")


def run(ctx: RunContext) -> dict:
    df = load_creators(resolve_project_path(ctx.config["data"]))
    write_csv(ctx.artifact("columns.csv"), describe_columns(df))
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_numeric_columns": len(numeric_columns(df)),
        "n_missing_cells": int(df.isna().sum().sum()),
    }


EXPERIMENT = Experiment(
    kind="eda",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
