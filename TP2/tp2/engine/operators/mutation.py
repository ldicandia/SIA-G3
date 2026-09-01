"""Mutation operators and per-gene-kind alleles modification.

Provides the four cátedra mutation scopes:
- gene: single gene mutation (MUT-01)
- multigen_limited: random count in [1, M] candidate genes (MUT-02)
- multigen_uniform: each gene independently with probability Pm (MUT-03)
- complete: with probability Pm, every gene mutates (MUT-04)

All four scopes support the Michalewicz non-uniform schedule (MUT-06) and
per-kind sigma overrides. Repair happens exclusively here via reflect (REP-03).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tp2.engine.genome import ACTIVE, GENES_PER_TRIANGLE, bounds_for, reflect, sigma_for
from .registry import MUTATION

__all__ = [
    "make_gene",
    "make_multigen_limited",
    "make_multigen_uniform",
    "make_complete",
    "apply_gene_mutation",
]

# Literature-derived starting points, not settled -- see STATE.md
DEFAULT_SIGMA_KIND = {
    "coordinate": 0.05,
    "color": 0.08,
    "alpha": 0.05,
}


def _validate_common_params(probability: float, schedule: str, b: float, sigma: dict | None = None) -> tuple[float, str, float, np.ndarray]:
    from tp2.engine.config import ConfigError

    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not (0.0 <= probability <= 1.0):
        raise ConfigError(f"mutation probability must be in [0, 1], got {probability!r}")
    if schedule not in ("uniform", "non_uniform"):
        raise ConfigError(f"mutation schedule must be 'uniform' or 'non_uniform', got {schedule!r}")
    if isinstance(b, bool) or not isinstance(b, (int, float)) or b <= 0:
        raise ConfigError(f"mutation schedule shape b must be positive, got {b!r}")

    # Build per-triangle sigma array
    per_triangle = np.array(
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.08, 0.05, 0.0],
        dtype=np.float32,
    )
    if sigma is not None:
        if not isinstance(sigma, dict):
            raise ConfigError(f"mutation sigma must be a dict, got {sigma!r}")
        for k, v in sigma.items():
            if k not in DEFAULT_SIGMA_KIND:
                raise ConfigError(f"unknown sigma override key {k!r}, expected one of {list(DEFAULT_SIGMA_KIND)}")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                raise ConfigError(f"mutation sigma for {k!r} must be positive, got {v!r}")

        coord_s = float(sigma.get("coordinate", DEFAULT_SIGMA_KIND["coordinate"]))
        color_s = float(sigma.get("color", DEFAULT_SIGMA_KIND["color"]))
        alpha_s = float(sigma.get("alpha", DEFAULT_SIGMA_KIND["alpha"]))

        per_triangle[0:6] = coord_s
        per_triangle[6:9] = color_s
        per_triangle[9] = alpha_s

    return float(probability), schedule, float(b), per_triangle


def _get_generation_progress(ctx: Any, gen_arg: int | None, max_gen_arg: int | None) -> float:
    """Extract generation and compute clamped progress = min(gen / max_gen, 1.0).

    Requires either `ctx` (with `.generation`/`.max_generations`, e.g. the
    engine's `GenerationContext`) or an explicit `generation` plus
    `max_generations` -- there is no hidden internal call counter and no
    invented default horizon. Before this fix, an unwired caller silently
    fell back to a per-*call* counter capped at a hardcoded
    `max_generations=100`; since mutation runs once per offspring (not once
    per generation), that counter saturated `progress` to `1.0` within a
    handful of real generations and made `schedule: "non_uniform"` a
    permanent no-op for the rest of the run (REVIEW.md CR-01). Raising here
    instead surfaces the missing wiring immediately, rather than silently
    mis-scheduling the Michalewicz decay.
    """
    if ctx is not None and hasattr(ctx, "generation") and hasattr(ctx, "max_generations"):
        gen = ctx.generation
        max_gen = ctx.max_generations
    elif gen_arg is not None:
        if max_gen_arg is None:
            raise ValueError(
                "non_uniform mutation schedule needs max_generations when generation is "
                "passed without ctx -- pass max_generations explicitly or supply a ctx with "
                "generation/max_generations set"
            )
        gen = gen_arg
        max_gen = max_gen_arg
    else:
        raise ValueError(
            "non_uniform mutation schedule needs a ctx (with .generation/.max_generations) "
            "or explicit generation/max_generations arguments -- got neither"
        )

    if max_gen <= 0:
        return 1.0
    return float(min(max(gen / max_gen, 0.0), 1.0))


def apply_gene_mutation(
    child: np.ndarray,
    locus: int,
    schedule: str,
    b: float,
    progress: float,
    sigmas: np.ndarray,
    bounds: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Mutate a single allele in place at locus with per-gene-kind behavior."""
    if locus % GENES_PER_TRIANGLE == ACTIVE:
        # Active flag is exempt from schedule: always binary flip with no fixed point
        child[locus] = 0.0 if child[locus] >= 0.5 else 1.0
        return

    lo = bounds[locus, 0]
    hi = bounds[locus, 1]

    if schedule == "non_uniform":
        # Michalewicz non-uniform schedule: Delta(t, y) = y * (1 - r^((1 - progress)^b))
        r = rng.random()
        exponent = (1.0 - progress) ** b
        delta_factor = 1.0 - (r ** exponent)
        if rng.random() < 0.5:
            delta = (hi - child[locus]) * delta_factor
            child[locus] += delta
        else:
            delta = (child[locus] - lo) * delta_factor
            child[locus] -= delta
    else:
        # Uniform schedule: gaussian step with locus-specific sigma
        child[locus] += rng.normal(0.0, sigmas[locus])

    child[locus:locus + 1] = reflect(child[locus:locus + 1], lo, hi)


@MUTATION.register("gene")
def make_gene(
    probability: float = 1.0,
    schedule: str = "uniform",
    b: float = 5.0,
    sigma: dict | None = None,
):
    """MUT-01: Single gene mutation."""
    prob_val, sched_val, b_val, per_tri_sigma = _validate_common_params(probability, schedule, b, sigma)

    def mutate(
        genes: np.ndarray,
        rng: np.random.Generator,
        ctx: Any = None,
        generation: int | None = None,
        max_generations: int | None = None,
    ) -> np.ndarray:
        child = np.asarray(genes, dtype=np.float32).copy()
        if child.size == 0 or rng.random() >= prob_val:
            return child

        triangles = child.size // GENES_PER_TRIANGLE
        sigmas = np.tile(per_tri_sigma, triangles)
        bounds = bounds_for(triangles)

        progress = _get_generation_progress(ctx, generation, max_generations) if sched_val == "non_uniform" else 0.0
        locus = int(rng.integers(child.size))
        apply_gene_mutation(child, locus, sched_val, b_val, progress, sigmas, bounds, rng)
        return child

    return mutate


@MUTATION.register("multigen_limited")
def make_multigen_limited(
    m: int,
    probability: float = 1.0,
    schedule: str = "uniform",
    b: float = 5.0,
    sigma: dict | None = None,
):
    """MUT-02: Limited multigene mutation picking a count in [1, M] candidates."""
    from tp2.engine.config import ConfigError

    if m is None or type(m) is not int or m < 1:
        raise ConfigError(f"multigen_limited m must be an integer >= 1, got {m!r}")

    prob_val, sched_val, b_val, per_tri_sigma = _validate_common_params(probability, schedule, b, sigma)
    m_val = m

    def mutate(
        genes: np.ndarray,
        rng: np.random.Generator,
        ctx: Any = None,
        generation: int | None = None,
        max_generations: int | None = None,
    ) -> np.ndarray:
        child = np.asarray(genes, dtype=np.float32).copy()
        if child.size == 0:
            return child

        triangles = child.size // GENES_PER_TRIANGLE
        sigmas = np.tile(per_tri_sigma, triangles)
        bounds = bounds_for(triangles)
        progress = _get_generation_progress(ctx, generation, max_generations) if sched_val == "non_uniform" else 0.0

        count_limit = min(m_val, child.size)
        # Exact integer count in [1, count_limit]
        c = int(rng.integers(1, count_limit + 1))
        candidate_loci = rng.choice(child.size, size=c, replace=False)

        for locus in candidate_loci:
            if rng.random() < prob_val:
                apply_gene_mutation(child, int(locus), sched_val, b_val, progress, sigmas, bounds, rng)

        return child

    return mutate


@MUTATION.register("multigen_uniform")
def make_multigen_uniform(
    probability: float = 1.0,
    schedule: str = "uniform",
    b: float = 5.0,
    sigma: dict | None = None,
):
    """MUT-03: Uniform multigene mutation where each locus independently has probability Pm."""
    prob_val, sched_val, b_val, per_tri_sigma = _validate_common_params(probability, schedule, b, sigma)

    def mutate(
        genes: np.ndarray,
        rng: np.random.Generator,
        ctx: Any = None,
        generation: int | None = None,
        max_generations: int | None = None,
    ) -> np.ndarray:
        child = np.asarray(genes, dtype=np.float32).copy()
        if child.size == 0 or prob_val == 0.0:
            return child

        triangles = child.size // GENES_PER_TRIANGLE
        sigmas = np.tile(per_tri_sigma, triangles)
        bounds = bounds_for(triangles)
        progress = _get_generation_progress(ctx, generation, max_generations) if sched_val == "non_uniform" else 0.0

        if prob_val == 1.0:
            for locus in range(child.size):
                apply_gene_mutation(child, locus, sched_val, b_val, progress, sigmas, bounds, rng)
        else:
            r = rng.random(child.size)
            mutating_loci = np.flatnonzero(r < prob_val)
            for locus in mutating_loci:
                apply_gene_mutation(child, int(locus), sched_val, b_val, progress, sigmas, bounds, rng)

        return child

    return mutate


@MUTATION.register("complete")
def make_complete(
    probability: float = 1.0,
    schedule: str = "uniform",
    b: float = 5.0,
    sigma: dict | None = None,
):
    """MUT-04: Complete mutation where every gene of the individual mutates with probability Pm."""
    prob_val, sched_val, b_val, per_tri_sigma = _validate_common_params(probability, schedule, b, sigma)

    def mutate(
        genes: np.ndarray,
        rng: np.random.Generator,
        ctx: Any = None,
        generation: int | None = None,
        max_generations: int | None = None,
    ) -> np.ndarray:
        child = np.asarray(genes, dtype=np.float32).copy()
        if child.size == 0 or rng.random() >= prob_val:
            return child

        triangles = child.size // GENES_PER_TRIANGLE
        sigmas = np.tile(per_tri_sigma, triangles)
        bounds = bounds_for(triangles)
        progress = _get_generation_progress(ctx, generation, max_generations) if sched_val == "non_uniform" else 0.0

        for locus in range(child.size):
            apply_gene_mutation(child, locus, sched_val, b_val, progress, sigmas, bounds, rng)

        return child

    return mutate
