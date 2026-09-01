"""(1+1) stochastic hill climber -- Roger Johansson's EvoLisa, reproduced honestly.

Population size ONE. No crossover. No selection pressure over a population.
No survival strategy. Each iteration mutates the single incumbent with this
project's own configured mutation operator and keeps the mutant only if its
fitness is STRICTLY better than the incumbent's -- accept-if-better,
reject-and-discard otherwise, and a rejected mutant is never re-evaluated.
This is not a genetic algorithm and must never be described, labeled, or
logged as one, or as "the baseline genetic algorithm" -- EvoLisa's own
author and this project's commenters explicitly reject that label for
exactly this algorithm, and mislabeling it here would misrepresent the
comparison EXP-05 exists to make.

Reuses `tp2.engine.fitness.Evaluator` and the SAME mutation operator a
multi-generation run would build from the identical `--config` file --
same method, same sigma table, same probability -- so an equal-render-budget
comparison against that run is honest rather than apples-to-oranges. Imports
nothing from `tp2.engine.operators.selection`, `.crossover`, or `.survival`;
importing any of those here would be a category error (see
`.planning/research/ARCHITECTURE.md`, "Internal boundaries").
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import PIL

from tp2.cli import resolve_seed
from tp2.engine.config import build_run_config, load_config
from tp2.engine.events import GenerationContext, GenerationEvent
from tp2.engine.fitness import Evaluator
from tp2.engine.genome import Population, active_count, random_population
from tp2.engine.stop import StopSet
from tp2.io.artifacts import prepare_run_dir, write_run_json, write_triangles_json
from tp2.io.images import load_target, save_png
from tp2.io.metrics import MetricsWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """A CLI configuration is inconsistent before the hill climber starts."""


@dataclass(frozen=True, slots=True)
class HillclimbResult:
    """Shaped like `tp2.engine.events.RunResult` where the fields overlap.

    It is its own type, not a subclass: a hill-climb result has no
    `best_genes`-vs-current-population distinction to preserve, since the
    population is always one individual.
    """

    genes: np.ndarray
    frame: np.ndarray
    fitness: float
    error: float
    iterations: int
    renders: int
    elapsed: float
    stop_reason: str


def run_hillclimb(
    evaluator: Evaluator,
    mutate: Callable,
    budget: int,
    rng: np.random.Generator,
    stop_set: StopSet,
    observer: Callable[[GenerationEvent], None] | None = None,
) -> HillclimbResult:
    """Run a (1+1) hill climb until one of `stop_set`'s conditions fires.

    Exactly one `evaluator.evaluate()` call happens per iteration, on top of
    the one call that scores the initial individual -- so
    `evaluator.renders == 1 + iterations` always, regardless of how many
    mutants were rejected along the way. A rejected mutant is discarded
    outright and never re-evaluated; this is the single most important
    accounting fact in this function, and it is what makes an equal-render-
    budget comparison against a genetic-algorithm run honest.

    `rng` is a single injected `numpy.random.Generator`, not a per-family
    stream bundle: `tp2/engine/rng.py`'s planned `make_streams(seed)` split
    was never built in this codebase (see STATE.md's Pending Todos item 3 --
    `tp2/engine/loop.py` and `tp2/cli.py` both inject one shared generator
    into every stochastic call), so this module follows that established
    convention rather than introducing a second, inconsistent RNG discipline
    for one baseline module alone.

    `observer`, when given, is called once per emitted `GenerationEvent`
    (generation 0's initial evaluation, then once per iteration) -- the same
    `obs(ev)` protocol `tp2/cli.py`'s observer list and `MetricsWriter`
    already satisfy, so a caller can feed a `MetricsWriter` directly.
    """
    started = time.perf_counter()
    genes = random_population(rng, 1, budget)[0]
    fitness, frame = evaluator.evaluate(genes)
    elapsed = time.perf_counter() - started
    horizon = stop_set.horizon

    def _snapshot(iteration: int, reason: str) -> GenerationEvent:
        genes_copy = genes.copy()
        genes_copy.flags.writeable = False
        return GenerationEvent(
            iteration, evaluator.renders, elapsed, fitness, 1.0 - fitness,
            # A population of one has no spread: mean and worst both equal
            # the (only) individual's fitness.
            fitness, fitness,
            # `diversity()` (tp2/engine/diversity.py) is undefined for N=1 --
            # a single-point sample has no meaningful stdev. 0.0 is the
            # honest "no diversity to measure" value; it is never computed by
            # calling diversity() with a 1-row array.
            0.0,
            active_count(genes), genes_copy, frame.copy(), reason,
        )

    pop = Population(genes.reshape(1, -1), np.array([fitness], dtype=np.float32))
    ctx = GenerationContext(0, horizon, evaluator.renders, elapsed, pop.fitness)
    reason = stop_set.check(ctx, pop)
    event = _snapshot(0, reason)
    if observer is not None:
        observer(event)
    if reason:
        return HillclimbResult(
            event.best_genes, event.best_frame, event.best_fitness, event.best_error,
            0, event.renders, event.elapsed, reason,
        )

    iteration = 0
    while True:
        iteration += 1
        ctx = GenerationContext(iteration, horizon, evaluator.renders, elapsed, np.array([fitness], dtype=np.float32))
        candidate = mutate(genes, rng, ctx=ctx)
        candidate_fitness, candidate_frame = evaluator.evaluate(candidate)
        # Accept STRICTLY (never `>=`): a tie never replaces the incumbent,
        # so a rejected mutant never needs a second evaluation and the
        # render-accounting invariant above stays exact.
        if candidate_fitness > fitness:
            genes, frame, fitness = candidate, candidate_frame, candidate_fitness
        elapsed = time.perf_counter() - started
        pop = Population(genes.reshape(1, -1), np.array([fitness], dtype=np.float32))
        ctx = GenerationContext(iteration, horizon, evaluator.renders, elapsed, pop.fitness)
        reason = stop_set.check(ctx, pop)
        event = _snapshot(iteration, reason)
        if observer is not None:
            observer(event)
        if reason:
            return HillclimbResult(
                event.best_genes, event.best_frame, event.best_fitness, event.best_error,
                iteration, event.renders, event.elapsed, reason,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a (1+1) stochastic hill climber -- population size one, no "
            "crossover, mutate-and-accept-if-better -- after Roger "
            "Johansson's EvoLisa. This is NOT a genetic algorithm: it reuses "
            "this project's own configured mutation operator only, at the "
            "same render budget, for an honest comparison against a "
            "genetic-algorithm run."
        ),
    )
    parser.add_argument("--image", required=True, type=Path, help="target image path")
    parser.add_argument("--triangles", type=int, default=30, help="triangle chromosome budget")
    parser.add_argument("--canvas", type=int, default=128, help="square render side in pixels")
    parser.add_argument(
        "--config", required=True, type=Path,
        help="JSON hyperparameter file (same shape tp2.cli's --config takes); only its mutation operator is used",
    )
    parser.add_argument("--seed", type=int, help="reproducibility seed; generated and archived if omitted")
    parser.add_argument("--out", type=Path, help="new output directory, relative to TP2 by default")
    parser.add_argument("--force", action="store_true", help="allow writing into an existing output directory")
    parser.add_argument("--allow-outside", action="store_true", help="permit --out outside this TP2 directory")
    parser.add_argument(
        "--max-renders", type=int, default=None,
        help=(
            "override --config's stop.max_generations horizon with an exact "
            "total render count (one initial render plus one per "
            "iteration): iterations == renders - 1. Set this to a "
            "genetic-algorithm run's total render count for an "
            "equal-budget comparison."
        ),
    )
    return parser


def _validate(args: argparse.Namespace) -> None:
    for name, value, low, high in (
        ("--triangles", args.triangles, 1, 5000),
        ("--canvas", args.canvas, 8, 1024),
    ):
        if not low <= value <= high:
            raise ConfigError(f"{name} must be in [{low}, {high}], got {value}")
    if not args.image.is_file():
        raise ConfigError(f"--image must name a readable file, got {args.image}")
    if not args.config.is_file():
        raise ConfigError(f"--config must name a readable file, got {args.config}")
    if args.max_renders is not None and args.max_renders < 2:
        raise ConfigError(
            f"--max-renders must be at least 2 (one initial render plus at least one iteration), got {args.max_renders}"
        )


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
        # The one `numpy.random.Generator` this module constructs. This
        # module is its own composition root, separate from `tp2/cli.py`
        # (see `tests/test_determinism.py`'s updated expectation, which now
        # names both files).
        rng = np.random.default_rng(seed)
        out = args.out or Path("runs") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{seed}-hillclimb"
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        run_dir = prepare_run_dir(out, PROJECT_ROOT, force=args.force, allow_outside=args.allow_outside)

        target = load_target(args.image, size)
        evaluator = Evaluator(target, size)

        config_data = load_config(args.config)
        if args.max_renders is not None:
            # iterations == renders - 1: horizon IS the iteration count
            # run_hillclimb performs before StopSet's max_generations
            # condition fires, so this is the exact override that gets a run
            # to within one iteration of a requested render budget. Forcing
            # stop.max_generations = True makes the override authoritative
            # even if the loaded config left that condition disabled.
            config_data = {
                **config_data,
                "horizon": args.max_renders - 1,
                "stop": {**config_data.get("stop", {}), "max_generations": True},
            }
        run_config = build_run_config(config_data)
        # ONLY the built mutation operator and stop set are used below.
        # `run_config`'s parents/replacement/crossover/survival callables are
        # an unavoidable side effect of build_run_config's validation -- they
        # are never called or referenced again below. Do not "helpfully" wire
        # any of them in; that would silently turn this into a genetic
        # algorithm.
        mutate = run_config.mutation
        stop_set = run_config.stop

        archive_config = {**config_data, "image": args.image, "canvas": args.canvas, "triangles": args.triangles}
        versions = {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__}
        # Written once up front for crash-safety, exactly like tp2/cli.py's
        # --config path; rewritten below once the run resolves.
        write_run_json(run_dir / "run.json", archive_config, seed, versions, _git_sha(), algorithm="hillclimb_1p1")

        with MetricsWriter(run_dir / "metrics.csv") as writer:
            result = run_hillclimb(evaluator, mutate, args.triangles, rng, stop_set, observer=writer)

        save_png(result.frame, run_dir / "best.png")
        write_triangles_json(run_dir / "triangles.json", result.genes, size)
        write_run_json(
            run_dir / "run.json", archive_config, seed, versions, _git_sha(),
            stop_reason=result.stop_reason, algorithm="hillclimb_1p1",
        )
    except (ConfigError, ValueError, OSError) as exc:
        build_parser().error(str(exc))
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
