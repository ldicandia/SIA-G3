"""Hand-written Kohonen SOM: BMU, neighborhood, schedules, grids, metrics and training invariants."""

from __future__ import annotations

import ast
import re
import types

import numpy as np
import pytest

from tp4.models import som as som_module
from tp4.models.som import (
    ADJACENCY_TOL,
    SomConfig,
    best_matching_unit,
    best_two_units,
    decay,
    distance_preservation,
    init_weights,
    make_grid,
    neighborhood,
    quantization_error,
    squared_distances,
    topographic_error,
    train_som,
)
from tp4.runs import PROJECT_ROOT


def _fake_cupy() -> types.ModuleType:
    """A stand-in `cupy` that delegates every array function to numpy."""
    fake = types.ModuleType("cupy")
    fake.__getattr__ = lambda name: getattr(np, name)  # PEP 562 module __getattr__
    return fake


def _data(n=30, d=4, seed=3):
    return np.random.default_rng(seed).normal(size=(n, d))


# --------------------------------------------------------------------------- BMU


def test_bmu_matches_brute_force_euclidean():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    W = rng.normal(size=(12, 5))
    bmu, second, dist = best_two_units(X, W)
    for i, x in enumerate(X):
        norms = [np.linalg.norm(x - w) for w in W]
        order = np.argsort(norms, kind="stable")
        assert bmu[i] == order[0] == best_matching_unit(W, x)
        assert second[i] == order[1]
        assert dist[i] == pytest.approx(norms[order[0]], abs=1e-12)


def test_bmu_tie_goes_to_lowest_index_and_other_is_second():
    W = np.array([[5.0, 5.0], [1.0, 1.0], [9.0, 9.0], [1.0, 1.0]])
    x = np.array([[1.2, 0.9]])
    assert best_matching_unit(W, x[0]) == 1
    bmu, second, _ = best_two_units(x, W)
    assert (bmu[0], second[0]) == (1, 3)


def test_identical_vectors_tie_exactly():
    W = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    d2 = squared_distances(np.array([[0.7, -0.4, 0.25]]), W)
    assert d2[0, 0] == d2[0, 1]


# --------------------------------------------------------------------------- neighborhood


def test_gaussian_is_one_at_winner_and_decreases_with_distance_and_shrinking_radius():
    d = np.array([0.0, 1.0, 2.0, 3.0])
    h = neighborhood(d, 2.0, "gaussian")
    assert h[0] == 1.0
    assert np.all(np.diff(h) < 0)
    assert np.all((h > 0) & (h <= 1))
    at_two = [float(neighborhood(np.array([2.0]), r, "gaussian")[0]) for r in (1.0, 2.0, 3.0)]
    assert at_two[0] < at_two[1] < at_two[2]


def test_step_includes_the_boundary_and_excludes_beyond():
    one = lambda d, r: float(neighborhood(np.array([d]), r, "step")[0])  # noqa: E731
    for d in (0.0, 0.5, 1.0, 1.0000000000000002):
        assert one(d, 1.0) == 1.0
    assert one(1.01, 1.0) == 0.0
    assert one(0.0, 0.25) == 1.0


def test_step_is_non_increasing_in_distance_and_radius():
    d = np.linspace(0, 4, 41)
    h = neighborhood(d, 1.5, "step")
    assert np.all(np.diff(h) <= 0)
    radii = [3.0, 2.0, 1.0, 0.5, 0.1]
    values = np.array([neighborhood(d, r, "step") for r in radii])
    assert np.all(np.diff(values, axis=0) <= 0)


@pytest.mark.parametrize("kind", ["gaussian", "step"])
def test_winner_always_has_full_influence_at_any_positive_radius(kind):
    for r in (1e-6, 0.1, 1.0, 50.0):
        assert neighborhood(np.array([0.0]), r, kind)[0] == 1.0


def test_neighborhood_rejects_bad_radius_and_kind():
    for r in (0.0, -1.0):
        with pytest.raises(ValueError):
            neighborhood(np.array([0.0, 1.0]), r, "gaussian")
    with pytest.raises(ValueError):
        neighborhood(np.array([0.0]), 1.0, "mexican")


# --------------------------------------------------------------------------- decay


@pytest.mark.parametrize("kind", ["exponential", "linear", "inverse"])
def test_decay_hits_start_and_end_and_never_increases(kind):
    total = 50
    values = [decay(0.5, 0.01, s, total, kind) for s in range(total)]
    assert values[0] == 0.5
    assert abs(values[-1] - 0.01) < 1e-12
    assert all(b <= a + 1e-15 for a, b in zip(values, values[1:]))
    assert decay(0.5, 0.01, 0, 1, kind) == 0.5


def test_decay_constant_and_errors():
    assert all(decay(0.3, 0.01, s, 20, "constant") == 0.3 for s in range(20))
    for kind in ("exponential", "inverse"):
        with pytest.raises(ValueError):
            decay(1.0, 0.0, 3, 10, kind)
        with pytest.raises(ValueError):
            decay(1.0, -0.1, 3, 10, kind)
    with pytest.raises(ValueError):
        decay(1.0, 0.1, 3, 10, "cosine")


# --------------------------------------------------------------------------- grid


def test_rect_grid_neighbors():
    g = make_grid(3, 4, "rect")
    assert len(g.neighbors(0)) == 2
    assert len(g.neighbors(5)) == 4
    assert list(g.neighbors(5)) == sorted(g.neighbors(5))


def test_hex_interior_neuron_has_six_neighbors_at_unit_distance():
    g = make_grid(4, 4, "hex")
    near = g.neighbors(5)
    assert len(near) == 6
    assert all(abs(g.distances[5, j] - 1.0) < 1e-9 for j in near)


@pytest.mark.parametrize("topology", ["rect", "hex"])
def test_one_by_n_ends_have_one_neighbor(topology):
    g = make_grid(1, 5, topology)
    counts = [len(g.neighbors(i)) for i in range(5)]
    assert counts == [1, 2, 2, 2, 1]


def test_grid_rejects_degenerate_sizes_and_unknown_topology():
    for rows, cols in ((1, 1), (0, 4), (4, 0)):
        with pytest.raises(ValueError):
            make_grid(rows, cols, "rect")
    with pytest.raises(ValueError):
        make_grid(3, 3, "tri")


def test_grid_distances_are_symmetric_and_index_is_row_major():
    g = make_grid(3, 5, "hex")
    assert np.array_equal(g.distances, g.distances.T)
    assert np.all(np.diag(g.distances) == 0)
    assert g.n_neurons == 15
    # neuron 7 is row 1, col 2 -> x = 2 + 0.5 (odd row), y = sqrt(3) / 2
    assert g.positions[7] == pytest.approx([2.5, np.sqrt(3) / 2])
    pairs = g.adjacency_pairs()
    assert pairs == sorted(pairs) and all(a < b for a, b in pairs)
    assert all(b in g.neighbors(a) for a, b in pairs)


# --------------------------------------------------------------------------- metrics


def test_quantization_error_is_zero_when_weights_contain_the_samples():
    X = _data(10, 3)
    assert quantization_error(X, X.copy()) == 0.0


def test_topographic_error_counts_non_adjacent_second_winners():
    g = make_grid(1, 3, "rect")
    W = np.array([[0.0], [10.0], [1.0]])
    # 0.4 -> bmu 0, second 2 (grid distance 2: not adjacent); 10 and 10.5 -> bmu 1, second 2 (adjacent)
    X = np.array([[0.4], [10.0], [10.5]])
    assert topographic_error(X, W, g) == pytest.approx(1 / 3)


def test_distance_preservation_perfect_and_undefined():
    g = make_grid(1, 5, "rect")
    X = np.arange(5, dtype=float)[:, None]
    assert distance_preservation(X, X.copy(), g) == pytest.approx(1.0, abs=1e-12)
    g2 = make_grid(1, 2, "rect")
    W = np.array([[0.0], [100.0]])
    assert np.isnan(distance_preservation(X, W, g2))


# --------------------------------------------------------------------------- init


def test_sample_init_uses_distinct_rows_or_rows_with_replacement():
    X = _data(10, 3)
    W = init_weights(X, 6, "sample", np.random.default_rng(1))
    rows = {tuple(r) for r in X}
    assert len({tuple(r) for r in W}) == 6 and all(tuple(r) in rows for r in W)
    big = init_weights(X, 25, "sample", np.random.default_rng(1))
    assert all(tuple(r) in rows for r in big)


def test_random_init_lies_in_the_feature_ranges():
    X = _data(30, 4)
    W = init_weights(X, 20, "random", np.random.default_rng(2))
    assert np.all(W >= X.min(axis=0)) and np.all(W <= X.max(axis=0))
    with pytest.raises(ValueError):
        init_weights(X, 4, "pca", np.random.default_rng(2))


# --------------------------------------------------------------------------- training


def _cfg(**kw):
    base = dict(rows=3, cols=3, epochs=6, eta0=0.5, eta_min=0.05, radius0=2.0, radius_min=1.0)
    base.update(kw)
    return SomConfig(**base)


def test_training_is_deterministic_per_seed():
    X = _data()
    a = train_som(X, _cfg(), np.random.default_rng(7))
    b = train_som(X, _cfg(), np.random.default_rng(7))
    c = train_som(X, _cfg(), np.random.default_rng(8))
    assert np.array_equal(a.weights, b.weights)
    assert all(np.array_equal(a.history[k], b.history[k]) for k in a.history)
    assert not np.array_equal(a.weights, c.weights)


def test_history_shape_and_schedule_endpoints():
    res = train_som(_data(), _cfg(epochs=8), np.random.default_rng(0))
    assert all(len(v) == 8 for v in res.history.values())
    assert list(res.history["epoch"]) == list(range(1, 9))
    assert np.all(np.diff(res.history["eta"]) <= 1e-15)
    assert np.all(np.diff(res.history["radius"]) <= 1e-15)
    assert abs(res.history["radius"][-1] - 1.0) < 1e-12
    assert abs(res.history["eta"][-1] - 0.05) < 1e-12


def test_constant_radius_when_min_equals_start():
    res = train_som(_data(), _cfg(radius0=2.0, radius_min=2.0), np.random.default_rng(0))
    assert np.all(res.history["radius"] == 2.0)
    res = train_som(_data(), _cfg(radius_decay="constant"), np.random.default_rng(0))
    assert np.all(res.history["radius"] == 2.0)


@pytest.mark.parametrize("init", ["sample", "random"])
@pytest.mark.parametrize("neigh", ["gaussian", "step"])
def test_weights_stay_in_the_convex_hull_of_the_data(init, neigh):
    X = _data()
    res = train_som(X, _cfg(init=init, neighborhood=neigh, eta0=1.0), np.random.default_rng(4))
    assert np.all(res.weights >= X.min(axis=0) - 1e-9)
    assert np.all(res.weights <= X.max(axis=0) + 1e-9)


def test_all_options_train_without_error():
    X = _data(12, 3)
    for topology in ("hex", "rect"):
        for eta_decay in ("exponential", "linear", "inverse", "constant"):
            res = train_som(
                X, _cfg(topology=topology, eta_decay=eta_decay, radius_decay="inverse", epochs=3),
                np.random.default_rng(0),
            )
            assert np.all(np.isfinite(res.weights))


def test_one_by_n_grid_trains():
    res = train_som(_data(), _cfg(rows=1, cols=6, radius0=2.0), np.random.default_rng(0))
    assert res.weights.shape == (6, 4)


def test_train_rejects_unknown_options():
    with pytest.raises(ValueError):
        train_som(_data(), _cfg(topology="tri"), np.random.default_rng(0))
    with pytest.raises(ValueError):
        train_som(_data(), _cfg(neighborhood="mexican"), np.random.default_rng(0))


def test_fake_cupy_gives_the_same_weights_as_numpy():
    X = _data()
    base = train_som(X, _cfg(), np.random.default_rng(5))
    fake = train_som(X, _cfg(), np.random.default_rng(5), xp=_fake_cupy())
    assert np.array_equal(base.weights, fake.weights)


# --------------------------------------------------------------------------- no SOM library


ALLOWED_TOP_LEVEL = {"__future__", "math", "dataclasses", "types", "typing", "collections", "numpy", "tp4"}
FORBIDDEN = re.compile(
    r"^\s*(import|from)\s+(minisom|somoclu|sompy|susi|simpsom|sklearn\.cluster|sklearn\.neighbors"
    r"|scipy\.cluster|scipy\.spatial)\b",
    re.MULTILINE,
)


def test_som_module_imports_only_numpy_and_stdlib():
    tree = ast.parse((PROJECT_ROOT / "tp4" / "models" / "som.py").read_text(encoding="utf-8"))
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            tops.add((node.module or "").split(".")[0])
    assert tops <= ALLOWED_TOP_LEVEL, tops - ALLOWED_TOP_LEVEL


def test_no_som_or_clustering_library_anywhere_in_tp4():
    offenders = []
    for path in (PROJECT_ROOT / "tp4").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if FORBIDDEN.search(text):
            offenders.append(str(path))
        if "sklearn" in text and path.name != "pca.py" and re.search(r"^\s*(import|from)\s+sklearn", text, re.M):
            offenders.append(f"{path} (sklearn outside tp4/analysis/pca.py)")
    assert not offenders, offenders


def test_adjacency_tolerance_is_a_module_constant():
    assert ADJACENCY_TOL == 1e-9 and som_module.TOPOLOGIES == ("hex", "rect")
