"""Parent/replacement selection methods from the cátedra.

The `coefficient` used by `blend` below is the selection-blend proportion
`A` -- an approved optional extra -- and is NOT the Brecha Generacional `G`
CATEDRA.md mandates for generational replacement (Phase 3's SUR-05). The two
are orthogonal and deliberately do not share vocabulary; see CATEDRA.md's
"Brecha Generacional G" section and its resolved A/B discrepancy note.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .registry import BLEND_MAX_DEPTH, SELECTION
from .sampling import sample_from_weights

__all__ = [
    "elite_counts",
    "ranking_pseudo_fitness",
    "boltzmann_exp_val",
    "make_elite",
    "make_random",
    "make_blend",
    "make_roulette",
    "make_universal",
    "make_ranking",
    "make_boltzmann",
    "make_tournament_deterministic",
    "make_tournament_probabilistic",
    "BLEND_MAX_DEPTH",
]


def elite_counts(k: int, n: int) -> np.ndarray:
    """Course elite multiplicity n(i) = ceil((K-i)/N), by fitness rank."""
    return np.maximum(0, np.ceil((k - np.arange(n)) / n).astype(int))


def ranking_pseudo_fitness(fitness: np.ndarray) -> np.ndarray:
    """Compute pseudo-fitness f'(i) = (N - rank(i)) / N for ranking selection.

    Rank is 1-based from 1 (best) to N (worst), sorted stably descending.
    The worst individual has f'(worst) = 0.0 and will never be selected.
    """
    n = fitness.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(-fitness, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    return (n - ranks) / float(n)


def boltzmann_exp_val(
    fitness: np.ndarray,
    t0: float,
    tc: float,
    k: float | None = None,
    generation: int = 0,
) -> np.ndarray:
    """Compute expected value ExpVal(i, g, T) = exp(f(i)/T) / <exp(f(x)/T)>_g.

    Uses max-subtraction to prevent numerical overflow at low temperatures.
    """
    k_val = k if k is not None else 0.05
    temp = tc + (t0 - tc) * math.exp(-k_val * generation)
    if temp <= 0:
        temp = 1e-12
    f = np.asarray(fitness, dtype=np.float64)
    f_max = np.max(f)
    w = np.exp((f - f_max) / temp)
    mean_w = np.mean(w)
    if mean_w <= 0:
        return np.ones_like(w)
    return w / mean_w


@SELECTION.register("elite")
def make_elite():
    def select(fitness: np.ndarray, k: int, _: np.random.Generator) -> np.ndarray:
        if k < 0 or fitness.size == 0:
            raise ValueError("selection requires non-empty population and non-negative count")
        ordered = np.argsort(-fitness, kind="stable")
        return np.repeat(ordered, elite_counts(k, fitness.size))
    return select


@SELECTION.register("random")
def make_random():
    """Muestreo Aleatorio (CATEDRA.md): picks at random, with no regard for fitness."""
    def select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        if k < 0 or fitness.size == 0:
            raise ValueError("selection requires non-empty population and non-negative count")
        return rng.integers(0, fitness.size, size=k)
    return select


@SELECTION.register("blend")
def make_blend(
    coefficient: float | None = None,
    method_1: dict | None = None,
    method_2: dict | None = None,
    depth: int = 0,
):
    """SEL-09: a coefficient blend of two selection methods."""
    from tp2.engine.config import ConfigError

    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not 0.0 <= coefficient <= 1.0:
        raise ConfigError(f"blend coefficient must be a number in [0.0, 1.0], got {coefficient!r}")
    if not isinstance(method_1, dict):
        raise ConfigError(f"blend method_1 must be an operator spec object, got {method_1!r}")
    if not isinstance(method_2, dict):
        raise ConfigError(f"blend method_2 must be an operator spec object, got {method_2!r}")

    select_1 = SELECTION.build(method_1, depth=depth + 1)
    select_2 = SELECTION.build(method_2, depth=depth + 1)

    def select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        n1 = int(round(coefficient * k))
        first = select_1(fitness, n1, rng)
        second = select_2(fitness, k - n1, rng)
        return np.concatenate([first, second])
    return select


@SELECTION.register("roulette")
def make_roulette():
    """SEL-02: Roulette selection over relative aptitude wheel."""
    def select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        if k < 0 or fitness.size == 0:
            raise ValueError("selection requires non-empty population and non-negative count")
        return sample_from_weights(fitness, k, mode="roulette", rng=rng)
    return select


@SELECTION.register("universal")
def make_universal():
    """SEL-03: Stochastic Universal Sampling (SUS) over relative aptitude wheel."""
    def select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        if k < 0 or fitness.size == 0:
            raise ValueError("selection requires non-empty population and non-negative count")
        return sample_from_weights(fitness, k, mode="sus", rng=rng)
    return select


@SELECTION.register("ranking")
def make_ranking():
    """SEL-04: Ranking selection running roulette on pseudo-fitness (N - rank(i)) / N."""
    def select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        if k < 0 or fitness.size == 0:
            raise ValueError("selection requires non-empty population and non-negative count")
        pseudo_fitness = ranking_pseudo_fitness(fitness)
        return sample_from_weights(pseudo_fitness, k, mode="roulette", rng=rng)
    return select


@SELECTION.register("boltzmann")
def make_boltzmann(
    t0: float | None = None,
    tc: float | None = None,
    k: float | None = None,
):
    """SEL-05: Boltzmann selection with exponential temperature schedule."""
    from tp2.engine.config import ConfigError

    if t0 is None or isinstance(t0, bool) or not isinstance(t0, (int, float)) or t0 <= 0:
        raise ConfigError(f"boltzmann t0 must be a number > 0, got {t0!r}")
    if tc is None or isinstance(tc, bool) or not isinstance(tc, (int, float)) or tc <= 0:
        raise ConfigError(f"boltzmann tc must be a number > 0, got {tc!r}")
    if k is not None and (isinstance(k, bool) or not isinstance(k, (int, float)) or k < 0):
        raise ConfigError(f"boltzmann k must be a non-negative number, got {k!r}")

    t0_val = float(t0)
    tc_val = float(tc)
    k_val = float(k) if k is not None else None

    # Counter tracking generation invocations
    gen_counter = [0]

    def select(fitness: np.ndarray, count: int, rng: np.random.Generator, generation: int | None = None) -> np.ndarray:
        if count < 0 or fitness.size == 0:
            raise ValueError("selection requires non-empty population and non-negative count")
        t = gen_counter[0] if generation is None else generation
        gen_counter[0] += 1

        exp_val = boltzmann_exp_val(fitness, t0_val, tc_val, k_val, generation=t)
        return sample_from_weights(exp_val, count, mode="roulette", rng=rng)

    return select


@SELECTION.register("tournament_deterministic")
def make_tournament_deterministic(m: int | None = None):
    """SEL-06: Deterministic tournament picking the fittest of M distinct candidates."""
    from tp2.engine.config import ConfigError

    if m is None or type(m) is not int or m < 1:
        raise ConfigError(f"tournament_deterministic m must be an integer >= 1, got {m!r}")

    m_val = m

    def select(fitness: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        n = fitness.size
        if count < 0 or n == 0:
            raise ValueError("selection requires non-empty population and non-negative count")
        if count == 0:
            return np.empty(0, dtype=int)

        contest_m = min(m_val, n)
        winners = np.empty(count, dtype=int)
        for i in range(count):
            candidates = rng.choice(n, size=contest_m, replace=False)
            winners[i] = candidates[np.argmax(fitness[candidates])]
        return winners

    return select


@SELECTION.register("tournament_probabilistic")
def make_tournament_probabilistic(threshold: float | None = None):
    """SEL-07: Probabilistic tournament (M=2) picking fitter if r < Threshold, else less fit."""
    from tp2.engine.config import ConfigError

    if (
        threshold is None
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not (0.5 <= threshold <= 1.0)
    ):
        raise ConfigError(f"tournament_probabilistic threshold must be in [0.5, 1.0], got {threshold!r}")

    thresh_val = float(threshold)

    def select(fitness: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        n = fitness.size
        if count < 0 or n == 0:
            raise ValueError("selection requires non-empty population and non-negative count")
        if count == 0:
            return np.empty(0, dtype=int)

        if n == 1:
            return np.zeros(count, dtype=int)

        winners = np.empty(count, dtype=int)
        for i in range(count):
            c1, c2 = rng.choice(n, size=2, replace=False)
            if fitness[c1] >= fitness[c2]:
                fitter, less_fit = c1, c2
            else:
                fitter, less_fit = c2, c1

            r = rng.random()
            # CATEDRA requirement: strict r < Threshold takes fitter, otherwise LESS FIT
            if r < thresh_val:
                winners[i] = fitter
            else:
                winners[i] = less_fit
        return winners

    return select
