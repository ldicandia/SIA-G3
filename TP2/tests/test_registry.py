"""Registry seam contracts.

Phase 3 adds roughly twenty entries across the four registries. This file is
what tells the author whether a new one is actually reachable: importing the
operators package must populate every registry with no explicit registration
call, and every reported name must build from a minimal spec.
"""

from __future__ import annotations

import numpy as np
import pytest

from tp2.engine.config import ConfigError
from tp2.engine.operators import registry  # noqa: F401 -- import alone populates all four registries
from tp2.engine.operators.registry import CROSSOVER, MUTATION, SELECTION, SURVIVAL

REGISTRIES = {"selection": SELECTION, "crossover": CROSSOVER, "mutation": MUTATION, "survival": SURVIVAL}


def test_importing_operators_package_populates_all_four_registries() -> None:
    for kind, reg in REGISTRIES.items():
        assert reg.names(), f"{kind} registry is empty after importing tp2.engine.operators"


def test_every_registered_name_builds_from_a_minimal_spec() -> None:
    replacement = SELECTION.build({"method": "elite"})
    minimal = {
        "selection": {
            "elite": {"method": "elite"},
            "random": {"method": "random"},
            "blend": {
                "method": "blend", "coefficient": 0.5,
                "method_1": {"method": "elite"}, "method_2": {"method": "random"},
            },
        },
        "crossover": {"one_point": {"method": "one_point", "boundary": "gene"}},
        "mutation": {"gene": {"method": "gene", "probability": 0.5}},
        "survival": {"additive": {"method": "additive", "replacement": replacement}},
    }
    for kind, reg in REGISTRIES.items():
        names = set(reg.names())
        assert names == set(minimal[kind]), (
            f"{kind} registry names {names} do not match this test's minimal-spec map "
            f"{set(minimal[kind])} -- a name was added without a buildable spec here"
        )
        for name in names:
            reg.build(minimal[kind][name])


def test_unknown_name_raises_naming_value_and_known_names() -> None:
    for reg in REGISTRIES.values():
        with pytest.raises(ConfigError) as excinfo:
            reg.build({"method": "bogus_operator_name"})
        message = str(excinfo.value)
        assert "bogus_operator_name" in message
        for name in reg.names():
            assert name in message


def test_blend_spec_builds_in_both_parents_and_replacement_slots() -> None:
    """SUR-04: the replacement slot accepts any selection the parents slot does."""
    blend_spec = {
        "method": "blend", "coefficient": 0.6,
        "method_1": {"method": "elite"}, "method_2": {"method": "random"},
    }
    parents = SELECTION.build(blend_spec)
    replacement = SELECTION.build(blend_spec)
    fitness = np.array([0.1, 0.5, 0.9, 0.2, 0.4, 0.7], dtype=np.float32)
    rng = np.random.default_rng(1)
    assert parents(fitness, 6, rng).shape == (6,)
    assert replacement(fitness, 6, rng).shape == (6,)


def test_blend_coefficient_boundaries_delegate_entirely_to_one_member() -> None:
    """A blend with coefficient 1.0/0.0 returns exactly its first/second member's
    result for a generator in the same (fresh) state -- pinning SEL-09's identity
    at the boundaries."""
    fitness = np.array([0.1, 0.9, 0.3, 0.7], dtype=np.float32)

    elite = SELECTION.build({"method": "elite"})
    random_method = SELECTION.build({"method": "random"})
    blend_1 = SELECTION.build({
        "method": "blend", "coefficient": 1.0,
        "method_1": {"method": "elite"}, "method_2": {"method": "random"},
    })
    blend_0 = SELECTION.build({
        "method": "blend", "coefficient": 0.0,
        "method_1": {"method": "elite"}, "method_2": {"method": "random"},
    })

    direct_elite = elite(fitness, 4, np.random.default_rng(42))
    via_blend_1 = blend_1(fitness, 4, np.random.default_rng(42))
    assert np.array_equal(direct_elite, via_blend_1)

    direct_random = random_method(fitness, 4, np.random.default_rng(42))
    via_blend_0 = blend_0(fitness, 4, np.random.default_rng(42))
    assert np.array_equal(direct_random, via_blend_0)


def test_blend_split_is_banker_rounded_and_pinned() -> None:
    """A-39: round(0.5 * 11) == 6 (banker's rounding), not a stochastic split."""
    calls: list[int] = []

    @SELECTION.register("_probe_for_blend_split_test")
    def make_probe():
        def select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
            calls.append(k)
            return np.zeros(k, dtype=int)
        return select

    try:
        blend = SELECTION.build({
            "method": "blend", "coefficient": 0.5,
            "method_1": {"method": "_probe_for_blend_split_test"},
            "method_2": {"method": "_probe_for_blend_split_test"},
        })
        blend(np.zeros(11, dtype=np.float32), 11, np.random.default_rng(0))
        assert calls == [6, 5], calls
    finally:
        del SELECTION._factories["_probe_for_blend_split_test"]
