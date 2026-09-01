"""Portable, inspectable artifacts for one run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tp2.engine.genome import A, ACTIVE, ACTIVE_THRESHOLD, B, GENES_PER_TRIANGLE, G, R, active_count


class RunDirError(ValueError):
    """An output-directory request would be unsafe or overwrite evidence."""


@dataclass(frozen=True, slots=True)
class Triangle:
    index: int
    points: list[list[int]]
    color: list[int]
    alpha: int
    active: bool


def triangles_from_genes(genes: np.ndarray, size: tuple[int, int]) -> list[Triangle]:
    width, height = size
    table = np.asarray(genes).reshape(-1, GENES_PER_TRIANGLE)
    triangles: list[Triangle] = []
    for index, row in enumerate(table):
        points = np.rint(row[:6].reshape(3, 2) * np.array([width, height])).astype(int).tolist()
        color = np.rint(row[[R, G, B]] * 255).astype(np.uint8).astype(int).tolist()
        triangles.append(
            Triangle(index, points, color, int(np.rint(row[A] * 255)), bool(row[ACTIVE] >= ACTIVE_THRESHOLD))
        )
    return triangles


def write_triangles_json(path: str | Path, genes: np.ndarray, size: tuple[int, int]) -> None:
    triangles = triangles_from_genes(genes, size)
    payload = {
        "canvas": {"width": size[0], "height": size[1]},
        "budget": len(triangles),
        "active_count": active_count(genes),
        "triangles": [asdict(triangle) for triangle in triangles],
    }
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _portable_path(path: str | Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return str(resolved)


def write_run_json(
    path: str | Path,
    effective_config: dict[str, Any],
    seed: int,
    versions: dict[str, str],
    git_sha: str | None = None,
    stop_reason: str | None = None,
) -> None:
    """Archive actual resolved settings, with home paths avoided when possible.

    `stop_reason` is optional so a caller may archive the config before a run
    starts (crash-safety) and again after it resolves. When present, it is a
    top-level field -- distinct from `config` -- naming which of the engine's
    stop conditions fired, or `viewer_closed` when an observer ended the run
    early (04-01: T-04-03, repudiation).
    """
    project_root = Path(__file__).resolve().parents[2]
    config = dict(effective_config)
    if "image" in config:
        config["image"] = _portable_path(config["image"], project_root)
    payload: dict[str, Any] = {"config": config, "seed": int(seed), "versions": versions}
    if git_sha is not None:
        payload["git_sha"] = git_sha
    if stop_reason is not None:
        payload["stop_reason"] = stop_reason
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def prepare_run_dir(
    out: str | Path,
    project_root: str | Path,
    *,
    force: bool = False,
    allow_outside: bool = False,
) -> Path:
    root = Path(project_root).resolve()
    destination = Path(out).resolve()
    if not allow_outside:
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise RunDirError(f"output directory is outside the project: {destination}") from exc
    if destination.exists() and any(destination.iterdir()) and not force:
        raise RunDirError(f"output directory already exists and is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination
