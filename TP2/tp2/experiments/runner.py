"""Process-pool matrix driver: one OS process per (cell, seed) job.

Mirrors `tp2/cli.py`'s single-run composition-root sequence headlessly inside
each worker (`run_cell_seed`): build a `RunConfig` via the SAME
`tp2.engine.config.build_run_config` validation a single run uses, build one
`Evaluator`, construct one `Run`, drive `for ev in run:` feeding only a
`MetricsWriter`, then write the four artifacts. No `Viewer` is imported
anywhere in this file's import list -- the viewer is structurally absent from
this code path, not merely unflagged.

`derive_seed` builds a fresh `numpy.random.SeedSequence([base_entropy,
cell_index, replicate_index])` per job. This composes safely with
`np.random.default_rng(seed)` (the engine's own composition-root convention,
see `tp2/cli.py` and `tp2/baselines/hillclimber.py` -- `tp2/engine/rng.py`'s
planned per-family `make_streams(seed)` split was never built anywhere in
this codebase, confirmed absent by reading the filesystem directly, per
STATE.md's own outstanding Phase-2-debt note): the derived integer is fed to
`np.random.default_rng` exactly as if it were an ordinary `--seed` value, so
a cell run under the matrix is indistinguishable, from the engine's point of
view, from an ordinary single-run invocation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import PIL

from tp2.engine.config import build_run_config
from tp2.engine.fitness import Evaluator
from tp2.engine.loop import Run
from tp2.io.artifacts import prepare_run_dir, write_run_json, write_triangles_json
from tp2.io.images import load_target, save_png
from tp2.io.metrics import MetricsWriter

from .matrix import MatrixSpec, MatrixSpecError, apply_overrides, build_cells, load_matrix_spec

__all__ = [
    "derive_seed",
    "CellJob",
    "CellRunSummary",
    "run_cell_seed",
    "build_jobs",
    "run_matrix",
    "build_parser",
    "main",
]

# tp2/experiments/runner.py -> tp2/experiments -> tp2 -> TP2 (project root),
# matching tp2/cli.py's own PROJECT_ROOT convention one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# T-04-07: a planned run count above this cap is rejected before a single
# process spawns, unless --allow-large is explicitly passed.
RUN_COUNT_CAP = 2000


def _stable_cell_index(cell_id: str) -> int:
    """A deterministic integer derived purely from `cell_id`'s own content.

    Deliberately NOT a rank in a sorted list of the OTHER cell_ids present:
    a sorted-position index would still shift for every cell alphabetically
    after a newly-inserted arm, breaking the very stability this function
    exists to provide (see `build_jobs`'s own docstring for the worked
    counter-example). A cell's identity determines its own index and
    nothing else's presence or absence can change it.
    """
    digest = hashlib.sha256(cell_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def derive_seed(base_entropy: int, cell_index: int, replicate_index: int) -> int:
    """Per-cell-per-replicate seed, independent of iteration order.

    A pure function of three integers -- never derived as `base + i` (adjacent
    seeds can correlate); `np.random.SeedSequence`'s spawn-key mixing is the
    documented mechanism for independent, reproducible parallel streams.
    """
    seq = np.random.SeedSequence([base_entropy, cell_index, replicate_index])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True, slots=True)
class CellJob:
    cell_id: str
    cell_index: int
    replicate_index: int
    base_seed: int
    seed: int
    resolved_config: dict[str, Any]
    image_path: Path
    canvas: int
    triangle_budget: int
    out_dir: Path


@dataclass(frozen=True, slots=True)
class CellRunSummary:
    cell_id: str
    replicate_index: int
    ok: bool
    renders: int
    error: str | None


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_cell_seed(job: CellJob) -> CellRunSummary:
    """Module-level (picklable) process-pool worker.

    The ENTIRE body is wrapped in `try/except Exception`: on any failure this
    returns a failed `CellRunSummary` rather than letting the exception
    propagate out of `Pool.map`, which would otherwise abort the entire batch
    on the first failing cell (T-04-09) -- exactly the silent-partial-failure
    this plan's first prohibition forbids.
    """
    try:
        config = build_run_config(job.resolved_config)
        rng = np.random.default_rng(job.seed)
        size = (job.canvas, job.canvas)
        target = load_target(job.image_path, size)
        evaluator = Evaluator(target, size)
        # Never force=True: a genuine collision here is exactly the failure
        # this plan's concurrency-safety truth exists to prevent (T-04-08).
        run_dir = prepare_run_dir(job.out_dir, PROJECT_ROOT, force=False)
        run = Run(config, evaluator, job.triangle_budget, rng)
        # CR-01: write metrics under a temporary name and only rename it to
        # the real `metrics.csv` after EVERY artifact for this run has been
        # written successfully. `MetricsWriter.write()` flushes after every
        # row, so a mid-loop crash (raised out of `for event in run`) would
        # otherwise leave a syntactically valid but truncated `metrics.csv`
        # behind -- indistinguishable from a complete replicate to anything
        # that later globs `seed*/metrics.csv` (tp2/experiments/aggregate.py,
        # scripts/generate_plots.py). By never letting a file literally named
        # `metrics.csv` exist until the run is fully done, a crashed seed's
        # directory has no `metrics.csv` at all, so downstream aggregation
        # fails loudly (FileNotFoundError naming this exact seed directory)
        # instead of silently folding a partial curve into a median/IQR
        # figure.
        tmp_metrics = run_dir / "metrics.csv.tmp"
        with MetricsWriter(tmp_metrics) as writer:
            for event in run:
                writer(event)
        result = run.result
        # WR-05: an explicit raise, not a bare `assert` -- this is a
        # defensive invariant check on library behavior, not a test
        # assertion, so it must still hold under `python -O` (assertions
        # stripped), rather than silently falling through to an
        # `AttributeError` on `result.stop_reason` below with a far less
        # informative message.
        if result is None:
            raise RuntimeError("Run.__iter__ must set run.result on its final yielded event")

        versions = {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__}
        # The resolved integer `children` (never the nominal ratio alone) is
        # already baked into job.resolved_config by apply_overrides -- and
        # config.effective carries it forward -- so it is traceable here.
        # cell_index/base_seed/replicate_index are archived alongside so
        # derive_seed(base_seed, cell_index, replicate_index) == seed can be
        # independently re-derived from this file alone.
        archive_config = {
            **config.effective,
            "image": str(job.image_path),
            "canvas": job.canvas,
            "triangles": job.triangle_budget,
            "cell_id": job.cell_id,
            "cell_index": job.cell_index,
            "replicate_index": job.replicate_index,
            "base_seed": job.base_seed,
        }
        write_run_json(
            run_dir / "run.json", archive_config, job.seed, versions, _git_sha(),
            stop_reason=result.stop_reason,
        )
        save_png(result.best_frame, run_dir / "best.png")
        write_triangles_json(run_dir / "triangles.json", result.best_genes, size)
        # Last step: only a truly-finished run (loop completed, run.json,
        # best.png, and triangles.json all written) ever becomes visible
        # under the name `metrics.csv`.
        tmp_metrics.replace(run_dir / "metrics.csv")
        return CellRunSummary(job.cell_id, job.replicate_index, ok=True, renders=evaluator.renders, error=None)
    except Exception as exc:  # noqa: BLE001 -- must never propagate out of Pool.map
        return CellRunSummary(job.cell_id, job.replicate_index, ok=False, renders=0, error=str(exc))


def build_jobs(spec: MatrixSpec) -> list[CellJob]:
    """Pure job-list construction -- no process spawned, nothing executed.

    Exposed as its own function (not inlined into `run_matrix`) so the seed-
    and path-uniqueness proofs at the real 75-job scale can be tested WITHOUT
    running a single one of them -- this plan's own stated verification
    requirement -- from one single source of truth shared with `run_matrix`
    itself, rather than a second, potentially-diverging reimplementation in
    the test suite.

    `cell_index` is a STABLE HASH of the cell's own `cell_id` string
    (`_stable_cell_index`), never a raw list position AND never a rank in a
    sorted order of the OTHER cell_ids present. A sorted-position index looks
    stable but is not: inserting a new arm whose name sorts alphabetically
    before an existing one (e.g. appending `"aardvark_arm"` after
    `"selection"`) would shift every later cell's rank and therefore its
    derived seed -- silently invalidating every already-run, already-cited
    experiment for cells that were never touched. Hashing the cell_id's own
    content instead means a cell's derived seed depends on nothing but its
    own identity: reordering the spec's `arms` mapping, or appending a
    brand-new arm anywhere in the JSON, can never change ANY existing cell's
    derived seed, regardless of where the new arm's name falls alphabetically.
    """
    cells = build_cells(spec)
    with spec.baseline_path.open(encoding="utf-8") as handle:
        baseline_raw = json.load(handle)

    jobs: list[CellJob] = []
    for cell in cells:
        cell_index = _stable_cell_index(cell.cell_id)
        resolved_config = apply_overrides(baseline_raw, cell.overrides)
        for replicate_index in range(spec.seeds):
            seed = derive_seed(spec.base_seed, cell_index, replicate_index)
            # Deterministic, collision-free path -- NEVER a timestamp.
            out_dir = spec.out_root / cell.cell_id / f"seed{replicate_index}"
            jobs.append(CellJob(
                cell_id=cell.cell_id,
                cell_index=cell_index,
                replicate_index=replicate_index,
                base_seed=spec.base_seed,
                seed=seed,
                resolved_config=resolved_config,
                image_path=spec.image_path,
                canvas=spec.canvas,
                triangle_budget=spec.triangles,
                out_dir=out_dir,
            ))
    return jobs


def run_matrix(spec: MatrixSpec, jobs: int) -> list[CellRunSummary]:
    job_list = build_jobs(spec)
    with multiprocessing.Pool(jobs) as pool:
        return pool.map(run_cell_seed, job_list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a configuration matrix, one process per cell*seed job."
    )
    parser.add_argument("--spec", required=True, type=Path, help="matrix spec JSON path")
    parser.add_argument("--out", required=True, type=Path, help="output root for every cell's run directories")
    parser.add_argument(
        "--jobs", type=int, default=min(os.cpu_count() or 4, 16),
        help="process pool size, validated in [1, 64]",
    )
    parser.add_argument(
        "--allow-large", action="store_true",
        help="permit a planned run count above the sanity cap of %d" % RUN_COUNT_CAP,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if not 1 <= args.jobs <= 64:
            raise MatrixSpecError(f"--jobs must be in [1, 64], got {args.jobs}")

        spec = load_matrix_spec(args.spec)
        out_root = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
        # WR-03: dataclasses.replace, not a manual re-listing of every
        # MatrixSpec field -- a field added to MatrixSpec later with a
        # default would otherwise silently fall back to that default here
        # instead of forwarding the loaded spec's actual value.
        spec = dataclasses.replace(spec, out_root=out_root)

        cells = build_cells(spec)
        total = len(cells) * spec.seeds
        if total > RUN_COUNT_CAP and not args.allow_large:
            raise MatrixSpecError(
                f"planned run count {total} exceeds the {RUN_COUNT_CAP}-run sanity cap; "
                "pass --allow-large to proceed"
            )
    except MatrixSpecError as exc:
        parser.error(str(exc))
        return 2  # unreachable: argparse's .error() calls sys.exit()

    results = run_matrix(spec, args.jobs)
    failed = [r for r in results if not r.ok]

    if failed:
        spec.out_root.mkdir(parents=True, exist_ok=True)
        failures_path = spec.out_root / "FAILURES.json"
        with failures_path.open("w", encoding="utf-8") as handle:
            json.dump(
                [
                    {"cell_id": r.cell_id, "replicate_index": r.replicate_index, "error": r.error}
                    for r in failed
                ],
                handle, indent=2,
            )
            handle.write("\n")
        print(
            f"MATRIX FAILED: {len(failed)}/{len(results)} jobs failed; see {failures_path}",
            file=sys.stderr,
        )
        return 1

    print(f"{len(cells)} cells x {spec.seeds} seeds = {len(results)} runs completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
