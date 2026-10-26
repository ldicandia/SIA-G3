"""Vary one base-config parameter at a time, preserving grid key and value order.

Each child changes one grid leaf. Other parameters retain their base values;
seed and plot settings come from the sweep. No combinations of grid keys are run.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import EXPERIMENT_NAME_RE, load_config
from ..errors import ConfigError
from ..parallel import run_parallel
from ..runs import Experiment, RunContext, resolve_project_path, write_csv

MAX_CHILDREN = 512
EXTRA_KEYS = frozenset({"base", "grid", "workers", "child_plots"})
PARAM_DEFAULTS: dict[str, Any] = {}

_STR_UNSAFE = re.compile(r"[^a-z0-9_-]")


def _slug(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value).replace(".", "p").replace("-", "m")
    if isinstance(value, str):
        return _STR_UNSAFE.sub("_", value.lower())
    raise ConfigError(
        f"sweep: grid values must be booleans, numbers or strings, got {value!r}"
    )


def child_name(key: str, value: Any) -> str:
    name = f"{key.rsplit('.', 1)[-1]}-{_slug(value)}"
    if not EXPERIMENT_NAME_RE.fullmatch(name):
        raise ConfigError(
            f"sweep: parameter {key!r} value {value!r} gives child name {name!r}, "
            f"which must match {EXPERIMENT_NAME_RE.pattern}"
        )
    return name


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = config
    for part in parents:
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        if not isinstance(child, dict):
            raise ConfigError(f"sweep: grid key {dotted!r} descends into non-object {part!r}")
        node = child
    node[leaf] = value


def expand_grid(
    base: Mapping[str, Any], grid: Mapping[str, Sequence[Any]], *, seed: int, plots: bool
) -> list[tuple[str, dict]]:
    """Apply each key/value separately to an independent copy of the base."""
    children: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for key, values in grid.items():
        for value in values:
            name = child_name(key, value)
            if name in seen:
                raise ConfigError(f"sweep: duplicate child name {name!r} (repeated or colliding values)")
            seen.add(name)
            child = copy.deepcopy(dict(base))
            child["seed"] = seed
            _set_dotted(child, key, value)
            child["experiment"] = name
            child["plots"] = plots
            children.append((name, child))
    return children


def _load_base(config: Mapping[str, Any]) -> dict[str, Any]:
    base = config.get("base")
    if not isinstance(base, str) or not base:
        raise ConfigError(f"sweep: base must be a non-empty string path, got {base!r}")
    loaded = load_config(resolve_project_path(base))
    if loaded.get("kind") == "sweep":
        raise ConfigError("sweep: base must not be a sweep (nested sweeps are not supported)")
    return loaded


def _check_grid(grid: Any) -> None:
    if not isinstance(grid, dict) or not grid:
        raise ConfigError("sweep: grid must be a non-empty object {config.key: [values]}")
    for key, values in grid.items():
        if not isinstance(key, str) or not key:
            raise ConfigError(f"sweep: grid keys must be non-empty strings, got {key!r}")
        if not isinstance(values, list) or not values:
            raise ConfigError(f"sweep: grid[{key!r}] must be a non-empty list of values")


def validate(config: dict) -> None:
    from .. import runner

    base = runner.validate_config(_load_base(config))
    _check_grid(config.get("grid"))
    workers = config.get("workers", 1)
    if type(workers) is not int or workers < 1:
        raise ConfigError(f"sweep: workers must be an integer >= 1, got {workers!r}")
    child_plots = config.get("child_plots", False)
    if type(child_plots) is not bool:
        raise ConfigError(f"sweep: child_plots must be a boolean, got {child_plots!r}")

    n_children = sum(len(values) for values in config["grid"].values())
    if n_children > MAX_CHILDREN:
        raise ConfigError(f"sweep: grid expands to {n_children} children (maximum {MAX_CHILDREN})")
    for _name, child in expand_grid(base, config["grid"], seed=config["seed"], plots=child_plots):
        runner.validate_config(child)


def _run_child(item: tuple[dict, str]) -> dict:
    """Worker entry point: runs one validated child config inside its own directory."""
    from .. import runner

    cfg, directory = item
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    return runner.execute(runner.validate_config(cfg), path)


def run(ctx: RunContext) -> dict:
    from ..runner import validate_config

    config = ctx.config
    base = validate_config(_load_base(config))
    grid = config["grid"]
    children = expand_grid(
        base, grid, seed=ctx.seed, plots=config.get("child_plots", False)
    )
    workers = config.get("workers", 1)
    items = [(cfg, str(ctx.run_dir / name)) for name, cfg in children]
    summaries = run_parallel(_run_child, items, workers)

    keys = list(grid)
    scalar_keys = sorted(
        {k for s in summaries for k, v in s.items() if not isinstance(v, list)}
    )
    rows = []
    variations = [(key, value) for key, values in grid.items() for value in values]
    for (name, cfg), summary, (parameter, value) in zip(children, summaries, variations, strict=True):
        row: dict[str, Any] = {"child": name}
        for key in keys:
            node: Any = cfg
            for part in key.split("."):
                node = node[part]
            row[key] = node
        row["varied_parameter"] = parameter
        row["parameter_value"] = value
        for k in scalar_keys:
            row[k] = summary.get(k)
        rows.append(row)
    table = pd.DataFrame(rows, columns=["child", *keys, "varied_parameter", "parameter_value", *scalar_keys])
    write_csv(ctx.artifact("sweep.csv"), table)
    summary: dict[str, Any] = {"n_children": len(children), "workers": workers, "grid_keys": keys}

    from . import get_experiment

    hook = get_experiment(base["kind"]).sweep_report
    if hook is not None:
        extra = hook(ctx, table)
        clash = sorted(set(extra) & set(summary))
        if clash:
            raise ConfigError(f"sweep: the {base['kind']} sweep report returned reserved summary keys {clash}")
        summary.update(extra)
    return summary


EXPERIMENT = Experiment(
    kind="sweep",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
