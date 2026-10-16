"""Experiment kind `sweep`: run a base experiment over a grid of config values.

Config keys (besides the common ones):
  base         path to the config being swept (relative to TP4/); must not be a sweep
  grid         {dotted.config.key: [values...]}; children are the cartesian product
               in config order (itertools.product), so sweep.csv order is stable
  workers      worker processes (int >= 1, default 1)
  child_plots  write PNGs in every child run (default false)

Each child is a full tp4 run (run.json + summary.json + artifacts) stored in
`<sweep run dir>/<child name>/`; `sweep.csv` has one row per child.

`tp4.runner` is imported only inside functions: the runner imports the
experiment registry, which imports this module (a top-level import would cycle).
"""

from __future__ import annotations

import copy
import itertools
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


def child_name(assignments: Sequence[tuple[str, Any]]) -> str:
    """Directory/experiment name of one grid point, e.g. `standardize-true__k-3`."""
    name = "__".join(f"{key.rsplit('.', 1)[-1]}-{_slug(value)}" for key, value in assignments)
    if not EXPERIMENT_NAME_RE.fullmatch(name):
        raise ConfigError(
            f"sweep: grid point {list(assignments)!r} gives child name {name!r}, which must match "
            f"{EXPERIMENT_NAME_RE.pattern}"
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
    """Cartesian product of `grid` (keys in config order) applied to deep copies of `base`."""
    keys = list(grid)
    children: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for values in itertools.product(*(grid[k] for k in keys)):
        assignments = list(zip(keys, values))
        name = child_name(assignments)
        if name in seen:
            raise ConfigError(f"sweep: duplicate grid point {name!r} (repeated values in the grid)")
        seen.add(name)
        child = copy.deepcopy(dict(base))
        for key, value in assignments:
            _set_dotted(child, key, value)
        child["experiment"] = name
        if "seed" not in grid:
            child["seed"] = seed
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

    base = _load_base(config)
    _check_grid(config.get("grid"))
    workers = config.get("workers", 1)
    if type(workers) is not int or workers < 1:
        raise ConfigError(f"sweep: workers must be an integer >= 1, got {workers!r}")
    child_plots = config.get("child_plots", False)
    if type(child_plots) is not bool:
        raise ConfigError(f"sweep: child_plots must be a boolean, got {child_plots!r}")

    n_children = 1
    for values in config["grid"].values():
        n_children *= len(values)
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
    config = ctx.config
    base = _load_base(config)
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
    for (name, cfg), summary in zip(children, summaries):
        row: dict[str, Any] = {"child": name}
        for key in keys:
            node: Any = cfg
            for part in key.split("."):
                node = node[part]
            row[key] = node
        for k in scalar_keys:
            row[k] = summary.get(k)
        rows.append(row)
    table = pd.DataFrame(rows, columns=["child", *keys, *scalar_keys])
    write_csv(ctx.artifact("sweep.csv"), table)
    summary: dict[str, Any] = {"n_children": len(children), "workers": workers, "grid_keys": keys}

    # Registry import is function-local for the same cycle reason as `runner` above. The hook is
    # a Python attribute of the registered Experiment, never a name read from a config (T-02-04).
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
