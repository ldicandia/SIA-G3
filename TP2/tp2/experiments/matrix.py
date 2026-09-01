"""Matrix-cell generation from a baseline config plus labeled overrides.

Pure module: no I/O beyond `json.load` for the spec file and the baseline it
references. `itertools.product` appears in exactly one place, `product_cells`
below -- passing it a single dimension yields a one-factor-at-a-time sweep,
passing it several yields a genuine cross product, and this module has no
opinion on which a caller uses.

This module has no knowledge of what any operator name means (it does not
know what "roulette" or "exclusive" mean). The one piece of population
arithmetic it IS allowed to know is `children_ratio` -> `children` resolution
in `apply_overrides`, confined to a single function so it stays easy to
audit.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping


class MatrixSpecError(ValueError):
    """A matrix spec JSON is malformed or fails validation."""


@dataclass(frozen=True, slots=True)
class MatrixCell:
    cell_id: str
    overrides: dict[str, Any]


def product_cells(dimensions: Mapping[str, list[tuple[str, dict]]]) -> Iterator[MatrixCell]:
    """Cross product over labeled override dimensions.

    `dimensions` maps a dimension name to a list of `(label, overrides)`
    pairs. Each combination `itertools.product` produces across the given
    dimension lists becomes one `MatrixCell`: its `cell_id` joins every
    dimension's own `"{name}-{label}"` with `-`, and its `overrides` is the
    combination's override dicts shallow-merged in dimension order (later
    dimensions' keys win -- whole-value replacement, never a recursive deep
    merge).

    Passing one dimension is a one-factor-at-a-time sweep; passing more than
    one is a genuine cross product. Both are equally supported -- this
    function does not know or care which a caller is doing.
    """
    names = list(dimensions.keys())
    option_lists = [dimensions[name] for name in names]
    for combo in itertools.product(*option_lists):
        parts: list[str] = []
        merged: dict[str, Any] = {}
        for name, (label, overrides) in zip(names, combo):
            parts.append(f"{name}-{label}")
            merged.update(overrides)
        yield MatrixCell(cell_id="-".join(parts), overrides=merged)


def apply_overrides(baseline_raw: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply whole-value-replacement overrides onto a copy of `baseline_raw`.

    Every key in `overrides` except `children_ratio` replaces the baseline's
    value for that key WHOLESALE -- a cell overriding `parents` supplies the
    complete `{"method": ..., ...}` object, or it does not override `parents`
    at all. Never a recursive per-key merge of a nested operator-spec dict.

    `children_ratio` is the one special key this function resolves:
    `children = round(children_ratio * baseline_raw["population"])`, using
    Python's own `round()` (banker's rounding, matching Pattern 3's blend
    coefficient precedent from 02-01/02-02). `children_ratio` itself never
    survives into the returned dict -- only the resolved `children` does.
    """
    result = dict(baseline_raw)
    for key, value in overrides.items():
        if key == "children_ratio":
            continue
        result[key] = value
    if "children_ratio" in overrides:
        ratio = overrides["children_ratio"]
        result["children"] = round(ratio * baseline_raw["population"])
    return result


@dataclass(frozen=True, slots=True)
class MatrixSpec:
    baseline_path: Path
    base_seed: int
    seeds: int
    arms: dict[str, dict[str, dict]]
    out_root: Path
    # Spec-level run parameters that live OUTSIDE tp2.engine.config.CONFIG_KEYS
    # (image/canvas/triangles are CLI-level concerns for a single `tp2.cli`
    # run -- `build_run_config` would reject them as unknown config keys).
    # No arm in this plan varies any of the three, so they are fixed once per
    # spec rather than per-cell.
    image_path: Path
    canvas: int
    triangles: int
    # Applied FIRST (before a cell's own arm override) in `build_cells`, so a
    # tracer spec and a full-scale spec share the exact same downstream
    # `apply_overrides` call and differ only in JSON content, never in code.
    scale_overrides: dict[str, Any] = field(default_factory=dict)


def _require_int(data: dict[str, Any], key: str, minimum: int) -> int:
    value = data.get(key)
    if isinstance(value, bool) or type(value) is not int or value < minimum:
        raise MatrixSpecError(f"{key} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_nonempty_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise MatrixSpecError(f"{key} must be a non-empty string, got {value!r}")
    return value


def load_matrix_spec(path: str | Path) -> MatrixSpec:
    spec_path = Path(path)
    try:
        with spec_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixSpecError(f"invalid matrix spec {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MatrixSpecError("matrix spec root must be an object")

    seeds = _require_int(data, "seeds", 1)
    base_seed = data.get("base_seed")
    if isinstance(base_seed, bool) or type(base_seed) is not int:
        raise MatrixSpecError(f"base_seed must be an integer, got {base_seed!r}")

    baseline = _require_nonempty_str(data, "baseline")
    out_root = _require_nonempty_str(data, "out_root")
    image = _require_nonempty_str(data, "image")
    canvas = _require_int(data, "canvas", 8)
    triangles = _require_int(data, "triangles", 1)

    arms = data.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise MatrixSpecError(f"arms must be a non-empty object, got {arms!r}")
    for arm_name, labeled in arms.items():
        if not isinstance(labeled, dict) or not labeled:
            raise MatrixSpecError(f"arm {arm_name!r} must be a non-empty object of label -> overrides")
        for label, overrides in labeled.items():
            if not isinstance(overrides, dict):
                raise MatrixSpecError(f"arm {arm_name!r} label {label!r} overrides must be an object")

    scale_overrides = data.get("scale_overrides", {})
    if not isinstance(scale_overrides, dict):
        raise MatrixSpecError(f"scale_overrides must be an object, got {scale_overrides!r}")

    # Resolved relative to the SPEC FILE's own directory, never the CWD, so a
    # spec can be invoked from anywhere.
    spec_dir = spec_path.resolve().parent
    return MatrixSpec(
        baseline_path=(spec_dir / baseline).resolve(),
        base_seed=base_seed,
        seeds=seeds,
        arms=arms,
        out_root=Path(out_root),
        image_path=(spec_dir / image).resolve(),
        canvas=canvas,
        triangles=triangles,
        scale_overrides=dict(scale_overrides),
    )


def build_cells(spec: MatrixSpec) -> list[MatrixCell]:
    """Union the OFAT arms a matrix spec defines into one flat cell list.

    Each arm becomes exactly one dimension passed to `product_cells` -- one
    dimension per arm is what keeps a multi-arm spec one-factor-at-a-time
    even though `product_cells` itself is a genuine, general cross product.
    Because `product_cells` already embeds the dimension (arm) name in each
    cell's `cell_id`, cells from different arms can never collide by label
    alone -- no separate prefixing step is needed here.
    """
    try:
        with spec.baseline_path.open(encoding="utf-8") as handle:
            json.load(handle)  # validated eagerly so a bad baseline fails at build time, not run time
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixSpecError(f"invalid baseline {spec.baseline_path}: {exc}") from exc

    cells: list[MatrixCell] = []
    for arm_name, labeled in spec.arms.items():
        dimension = {arm_name: [(label, overrides) for label, overrides in labeled.items()]}
        for cell in product_cells(dimension):
            # scale_overrides first, the cell's own arm override layered on
            # top (plain dict merge, later key wins) -- identical to the
            # whole-value-replacement contract `apply_overrides` itself
            # applies downstream.
            combined = {**spec.scale_overrides, **cell.overrides}
            cells.append(MatrixCell(cell_id=cell.cell_id, overrides=combined))
    return cells
