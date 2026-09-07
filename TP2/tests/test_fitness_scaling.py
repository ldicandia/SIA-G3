"""Opt-in fitness scaling for the raw-fitness selection wheels.

`fitness = 1 - RMSE` sits on a large constant offset, so proportional
selection is nearly uniform. Scaling subtracts a baseline so the spread, not
the offset, drives the wheel. It is opt-in: CATEDRA.md defines Ruleta on raw
relative aptitude, so the default must not change any archived result.
"""

from __future__ import annotations

import numpy as np
import pytest

from tp2.engine.config import ConfigError
from tp2.engine.operators.registry import SELECTION
from tp2.engine.operators import selection as _selection  # noqa: F401  (registers)
from tp2.engine.operators.sampling import scale_fitness
from tp2.engine.operators.selection import boltzmann_exp_val

# A realistic population: fitness near 0.7 with a small spread, the regime
# that makes proportional selection nearly uniform in the first place.
POP = np.array([0.7161, 0.7050, 0.6890, 0.6720, 0.6580, 0.6301, 0.5491])


def test_none_is_exactly_identity():
    assert np.array_equal(scale_fitness(POP, "none"), POP.astype(np.float64))


def test_offset_zeroes_the_worst_and_preserves_differences():
    scaled = scale_fitness(POP, "offset")
    assert scaled.min() == 0.0
    assert np.allclose(np.diff(scaled), np.diff(POP.astype(np.float64)))


def test_scaling_raises_the_best_individuals_share_of_the_wheel():
    def share(w):
        w = np.asarray(w, dtype=np.float64)
        p = w / w.sum()
        return p.max() / p.mean()

    raw = share(scale_fitness(POP, "none"))
    offset = share(scale_fitness(POP, "offset"))
    sigma = share(scale_fitness(POP, "sigma"))

    assert raw < 1.2, "raw fitness on this offset is nearly uniform -- the whole problem"
    assert offset > raw
    assert sigma > raw


@pytest.mark.parametrize("mode", ["offset", "sigma"])
def test_a_uniform_population_stays_selectable(mode):
    """Every subtraction zeroes the worst; an all-equal population would
    collapse to an all-zero vector, which sample_from_weights rejects.
    """
    flat = np.full(5, 0.7)
    scaled = scale_fitness(flat, mode)
    assert scaled.sum() > 0


@pytest.mark.parametrize("mode", ["offset", "sigma"])
def test_a_lone_individual_stays_selectable(mode):
    assert scale_fitness(np.array([0.7]), mode).sum() > 0


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        scale_fitness(POP, "not_a_mode")


@pytest.mark.parametrize("method", ["roulette", "universal"])
def test_default_build_is_unscaled_and_matches_an_explicit_none(method):
    """The default must leave every archived run reproducible."""
    a = SELECTION.build({"method": method})
    b = SELECTION.build({"method": method, "scaling": "none"})
    ia = a(POP, 20, np.random.default_rng(7))
    ib = b(POP, 20, np.random.default_rng(7))
    assert np.array_equal(ia, ib)


@pytest.mark.parametrize("method", ["roulette", "universal"])
def test_invalid_scaling_is_rejected_at_build_time(method):
    with pytest.raises(ConfigError):
        SELECTION.build({"method": method, "scaling": "nope"})
    with pytest.raises(ConfigError):
        SELECTION.build({"method": method, "scaling": "sigma", "sigma_c": 0})


@pytest.mark.parametrize("method", ["roulette", "universal"])
def test_scaled_selection_prefers_fitter_individuals_more_often(method):
    unscaled = SELECTION.build({"method": method})
    scaled = SELECTION.build({"method": method, "scaling": "offset"})
    rng = np.random.default_rng(0)
    mean_unscaled = np.mean([POP[unscaled(POP, 40, rng)].mean() for _ in range(400)])
    mean_scaled = np.mean([POP[scaled(POP, 40, rng)].mean() for _ in range(400)])
    assert mean_scaled > mean_unscaled


def test_additive_scaling_is_a_no_op_for_boltzmann():
    """Why scaling is deliberately NOT wired into boltzmann: exp((f-c)/T)
    factors into exp(f/T)*exp(-c/T), and ExpVal's division by the population
    mean cancels the constant. A knob there would silently do nothing.
    """
    for t in (5.0, 0.5, 0.002):
        raw = boltzmann_exp_val(POP, t, t, 0.0, 0)
        shifted = boltzmann_exp_val(scale_fitness(POP, "offset"), t, t, 0.0, 0)
        assert np.allclose(raw, shifted, atol=0, rtol=0)
