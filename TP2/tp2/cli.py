"""Portable command-line composition root for the initial TP2 slice."""

from __future__ import annotations

import argparse
import contextlib
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import PIL

from .engine.events import GenerationEvent, RunResult
from .engine.config import build_run_config, load_config
from .engine.fitness import Evaluator
from .engine.genome import Population, active_count, random_population
from .engine.loop import Run
from .io.artifacts import prepare_run_dir, write_run_json, write_triangles_json
from .io.images import load_target, save_png
from .io.metrics import MetricsWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    """A CLI configuration is inconsistent before the engine can allocate."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approximate an image with translucent triangles.")
    parser.add_argument("--image", required=True, type=Path, help="target image path")
    parser.add_argument("--triangles", type=int, default=30, help="triangle chromosome budget")
    parser.add_argument("--population", type=int, default=8, help="population size")
    parser.add_argument("--canvas", type=int, default=128, help="square render side in pixels")
    parser.add_argument("--seed", type=int, help="reproducibility seed; generated and archived if omitted")
    parser.add_argument("--config", type=Path, help="JSON GA configuration; enables the multi-generation run")
    parser.add_argument("--notebook", action="store_true", help="show the best frame live in Jupyter or Google Colab")
    parser.add_argument("--notebook-every", type=int, default=1, help="update notebook view every N generations")
    parser.add_argument("--viewer", action="store_true", help="show a live pygame window of the evolving best frame")
    parser.add_argument("--viewer-scale", type=int, default=4, help="pixel scale factor for the live viewer window")
    parser.add_argument("--viewer-every", type=int, default=1, help="redraw the live viewer every N generations")
    parser.add_argument("--out", type=Path, help="new output directory, relative to TP2 by default")
    parser.add_argument("--force", action="store_true", help="allow writing into an existing output directory")
    parser.add_argument("--allow-outside", action="store_true", help="permit --out outside this TP2 directory")
    return parser


def _validate(args: argparse.Namespace) -> None:
    for name, value, low, high in (
        ("--triangles", args.triangles, 1, 5000),
        ("--population", args.population, 1, 10000),
        ("--canvas", args.canvas, 8, 1024),
    ):
        if not low <= value <= high:
            raise ConfigError(f"{name} must be in [{low}, {high}], got {value}")
    if not args.image.is_file():
        raise ConfigError(f"--image must name a readable file, got {args.image}")
    if args.notebook_every < 1:
        raise ConfigError(f"--notebook-every must be at least 1, got {args.notebook_every}")
    # Validated unconditionally -- regardless of whether --viewer was passed --
    # so a typo is caught even on a headless run that happens to also pass a
    # bad --viewer-scale/--viewer-every.
    if not 1 <= args.viewer_scale <= 16:
        raise ConfigError(f"--viewer-scale must be in [1, 16], got {args.viewer_scale}")
    if not 1 <= args.viewer_every <= 10000:
        raise ConfigError(f"--viewer-every must be in [1, 10000], got {args.viewer_every}")


def resolve_seed(args: argparse.Namespace) -> int:
    if args.seed is not None:
        return args.seed
    return int.from_bytes(__import__("secrets").token_bytes(8), "big") >> 1


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate(args)
        seed = resolve_seed(args)
        size = (args.canvas, args.canvas)
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        out = args.out or Path("runs") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{seed}"
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        run_dir = prepare_run_dir(out, PROJECT_ROOT, force=args.force, allow_outside=args.allow_outside)
        target = load_target(args.image, size)
        evaluator = Evaluator(target, size)
        if args.config:
            config_data = load_config(args.config)
            config = build_run_config(config_data)
            archive_config = {**config.effective, "image": args.image, "canvas": args.canvas, "triangles": args.triangles}
            versions = {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__}
            # Written once up front for crash-safety: even a run that never
            # reaches a stop condition leaves a traceable config+seed record.
            # Rewritten below with the resolved stop_reason once the run ends.
            write_run_json(run_dir / "run.json", archive_config, seed, versions, _git_sha())

            observer = None
            if args.notebook:
                from .ui.notebook import NotebookProgress
                observer = NotebookProgress(args.notebook_every)

            # Generic observer fan-out (ARCHITECTURE.md Pattern 4): the
            # engine calls nothing and knows none of these consumers exist.
            observers = [MetricsWriter(run_dir / "metrics.csv")]
            if args.viewer:
                # Import is local AND conditional so a headless invocation
                # never imports tp2.ui, let alone pygame.
                from tp2.ui.viewer import Viewer
                observers.append(Viewer(scale=args.viewer_scale, every=args.viewer_every))

            run = Run(config, evaluator, args.triangles, rng)
            last_event: GenerationEvent | None = None
            with contextlib.ExitStack() as stack:
                for obs in observers:
                    stack.enter_context(obs)
                for event in run:
                    last_event = event
                    for obs in observers:
                        obs(event)
                    if observer:
                        observer(event)
                    # Only break early on an observer's request when the
                    # engine itself has NOT already decided to stop on this
                    # event: if `event.stop_reason` is already set, letting
                    # the `for` loop finish naturally (one more `next()`)
                    # is what lets `Run` populate `run.result` with the
                    # honest hall-of-fame best -- breaking here instead would
                    # mislabel a naturally-completed run as `viewer_closed`.
                    if not event.stop_reason and any(getattr(obs, "should_stop", False) for obs in observers):
                        break  # plain break -- never an exception into the engine

            if run.result is not None:
                result = run.result
            else:
                # The loop was broken early by an observer (e.g. the viewer's
                # window was closed) before the engine's own stop condition
                # fired. `last_event` is guaranteed set: `for event in run`
                # always yields the generation-0 event before any observer
                # has a chance to request a stop.
                #
                # Honest limitation: this reports the LAST YIELDED event's
                # current-population best, not the engine's internal
                # hall-of-fame-best-ever (only reachable through `run.result`,
                # which is precisely absent here). Under exclusive survival
                # this can, rarely, be a worse individual than one seen
                # earlier in the same aborted run. This approximation applies
                # only to the interactive-abort path; a naturally-completed
                # run is unaffected and always reports the true
                # hall-of-fame best via `run.result`.
                assert last_event is not None
                result = RunResult(
                    best_genes=last_event.best_genes,
                    best_frame=last_event.best_frame,
                    best_fitness=last_event.best_fitness,
                    best_error=last_event.best_error,
                    generations=last_event.generation,
                    renders=last_event.renders,
                    elapsed=last_event.elapsed,
                    stop_reason="viewer_closed",
                )

            save_png(result.best_frame, run_dir / "best.png")
            write_triangles_json(run_dir / "triangles.json", result.best_genes, size)
            write_run_json(run_dir / "run.json", archive_config, seed, versions, _git_sha(), stop_reason=result.stop_reason)
        else:
            genes = random_population(rng, args.population, args.triangles)
            fitness, frames = evaluator.evaluate_population(genes)
            population = Population(genes, fitness)
            winner_index = int(np.argmax(population.fitness))
            best_genes = population.genes[winner_index].copy()
            best_frame = frames[winner_index]
            best_fitness = float(population.fitness[winner_index])
            event = GenerationEvent(0, evaluator.renders, time.perf_counter() - started, best_fitness,
                                    1.0 - best_fitness, float(population.fitness.mean()), float(population.fitness.min()),
                                    float(population.genes.std(axis=0).mean()), active_count(best_genes), best_genes,
                                    best_frame, "slice0")
            save_png(best_frame, run_dir / "best.png")
            write_triangles_json(run_dir / "triangles.json", best_genes, size)
            write_run_json(run_dir / "run.json", {"image": args.image, "canvas": args.canvas, "triangles": args.triangles, "population": args.population}, seed,
                           {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__}, _git_sha())
            with MetricsWriter(run_dir / "metrics.csv") as writer:
                writer.write(event)
    except (ConfigError, ValueError, OSError) as exc:
        build_parser().error(str(exc))
    print(run_dir)
    return 0
