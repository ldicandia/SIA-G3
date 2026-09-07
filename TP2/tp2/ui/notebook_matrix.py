"""Live side-by-side matrix comparison of multiple GA runs in a notebook.

Steps several `Run` iterators in lockstep -- one generation at a time, round
robin across every cell -- and, every `every` generations, composites each
cell's current best frame into a single labeled grid image, redrawn in place
in a Jupyter/Colab output cell. The multi-cell analogue of
`tp2.ui.notebook.NotebookProgress`, built on the same cell-generation
machinery `tp2.experiments.matrix` already uses for the offline experiment
runner: a crossover-method x mutation-method (or any other two-dimension)
cross product built via `product_cells`, `apply_overrides` onto one baseline
config.

`build_configs` and `compose_grid` are plain functions with no IPython
dependency -- testable without a notebook. `NotebookMatrix` and `run_matrix`
below them require IPython, exactly like `NotebookProgress`.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
from PIL import Image, ImageDraw

from tp2.engine.config import RunConfig, build_run_config, load_config
from tp2.engine.events import GenerationEvent, RunResult
from tp2.engine.fitness import Evaluator
from tp2.engine.loop import Run
from tp2.experiments.matrix import MatrixCell, apply_overrides, product_cells
from tp2.io.images import load_target

CAPTION_BAND_PX = 16


def build_configs(
    baseline_raw: dict[str, Any], cells: list[MatrixCell]
) -> dict[str, RunConfig]:
    """One `RunConfig` per cell, each the baseline with that cell's overrides applied."""
    return {
        cell.cell_id: build_run_config(apply_overrides(baseline_raw, cell.overrides))
        for cell in cells
    }


def compose_grid(
    frames: Mapping[str, np.ndarray],
    cell_ids: list[list[str]],
    captions: Mapping[str, str],
    scale: int = 2,
) -> np.ndarray:
    """Tile `frames[cell_ids[r][c]]` into one row-major grid, upscaled by `scale`,
    with a one-line caption drawn under each tile.

    `cell_ids` fixes the grid's shape (rows x cols) independently of dict
    iteration order -- every row must be the same length, and every id it
    names must be a key in both `frames` and `captions`.
    """
    n_cols = len(cell_ids[0])
    if any(len(row) != n_cols for row in cell_ids):
        raise ValueError(
            "compose_grid: every row of cell_ids must have the same length"
        )
    sample = next(iter(frames.values()))
    h, w, _ = sample.shape
    cell_w, cell_h = w * scale, h * scale
    canvas = Image.new(
        "RGB",
        (cell_w * n_cols, (cell_h + CAPTION_BAND_PX) * len(cell_ids)),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(cell_ids):
        for c, cell_id in enumerate(row):
            tile = Image.fromarray(
                np.asarray(frames[cell_id], dtype=np.uint8), "RGB"
            ).resize((cell_w, cell_h), Image.NEAREST)
            x, y = c * cell_w, r * (cell_h + CAPTION_BAND_PX)
            canvas.paste(tile, (x, y))
            draw.text((x + 2, y + cell_h + 2), captions[cell_id], fill=(255, 255, 255))
    return np.asarray(canvas)


class NotebookMatrix:
    """Render a refreshed grid PNG for each round of generations in a notebook output cell."""

    def __init__(self, every: int = 1) -> None:
        if every < 1:
            raise ValueError("notebook matrix update interval must be at least 1")
        self.every = every
        try:
            from IPython.display import Image as DisplayImage, clear_output, display
        except ImportError as exc:
            raise RuntimeError(
                "notebook matrix view requires Jupyter, Colab, or another IPython notebook"
            ) from exc
        self._display_image = DisplayImage
        self._clear_output = clear_output
        self._display = display

    def show(self, generation: int, grid: np.ndarray, status_line: str) -> None:
        encoded = BytesIO()
        Image.fromarray(grid, "RGB").save(encoded, format="PNG")
        self._clear_output(wait=True)
        self._display(f"Generation {generation} · {status_line}")
        self._display(self._display_image(data=encoded.getvalue()))


def run_matrix(
    image: str | Path,
    triangles: int,
    canvas: int,
    baseline_config: str | Path,
    row_dimension: str,
    row_options: list[tuple[str, dict[str, Any]]],
    col_dimension: str,
    col_options: list[tuple[str, dict[str, Any]]],
    seed: int = 42,
    every: int = 5,
    scale: int = 2,
) -> dict[str, RunResult]:
    """Evolve every (row option, col option) combination side by side, live, in a notebook cell.

    `row_options`/`col_options` are `(label, overrides)` pairs -- the same
    shape `product_cells` expects, e.g.
    `[("one_point", {"crossover": {"method": "one_point", "boundary": "triangle"}}), ...]`
    for a crossover row and
    `[("gene", {"mutation": {"method": "gene", "probability": 0.1}}), ...]`
    for a mutation column. Every cell starts from the SAME seed (only its
    operators differ), so the grid is an honest operator comparison, not
    also a seed comparison.

    Runs are stepped in lockstep: one `next()` per active cell per round, so
    a cell that reaches its own stop condition early simply freezes (its
    last frame stays on screen) while the rest keep evolving. Returns each
    cell's final `RunResult`, keyed by `cell_id`.
    """
    size = (canvas, canvas)
    baseline_raw = load_config(baseline_config)
    dims = {row_dimension: row_options, col_dimension: col_options}
    cells = list(product_cells(dims))
    n_cols = len(col_options)
    cell_ids = [
        [cell.cell_id for cell in cells[r * n_cols : (r + 1) * n_cols]]
        for r in range(len(row_options))
    ]
    captions_prefix = {
        cell.cell_id: cell.cell_id.replace(f"{row_dimension}-", "").replace(
            f"-{col_dimension}-", " / "
        )
        for cell in cells
    }
    configs = build_configs(baseline_raw, cells)

    target = load_target(image, size)
    runs: dict[str, Run] = {}
    active: dict[str, Iterator[GenerationEvent]] = {}
    for cell_id, config in configs.items():
        evaluator = Evaluator(target, size)
        rng = np.random.default_rng(seed)
        run = Run(config, evaluator, triangles, rng)
        runs[cell_id] = run
        active[cell_id] = iter(run)

    last_events: dict[str, GenerationEvent] = {}
    observer = NotebookMatrix(every)
    generation = 0
    while active:
        for cell_id in list(active):
            try:
                last_events[cell_id] = next(active[cell_id])
            except StopIteration:
                del active[cell_id]
        if not last_events:
            break
        generation = max(ev.generation for ev in last_events.values())
        if generation % every == 0 or not active:
            frames = {cid: ev.best_frame for cid, ev in last_events.items()}
            captions = {
                cid: f"{captions_prefix[cid]} · fit {last_events[cid].best_fitness:.3f}"
                for cid in captions_prefix
            }
            grid = compose_grid(frames, cell_ids, captions, scale)
            status = " · ".join(
                f"{captions_prefix[cid]}: {ev.best_fitness:.4f}"
                for cid, ev in last_events.items()
            )
            observer.show(generation, grid, status)

    return {cell_id: run.result for cell_id, run in runs.items()}
