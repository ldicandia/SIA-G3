"""Hand-written Kohonen SOM: BMU, neighborhood, schedules, grids, metrics and training invariants."""

from __future__ import annotations

import ast
import re

import numpy as np
import pytest

from tp4.models import som as som_module
from tp4.models.som import (
    ADJACENCY_TOL,
    SomConfig,
    best_matching_unit,
    matching_units,
    decay,
    init_weights,
    make_grid,
    neighborhood,
    squared_distances,
    train_som,
)
from tp4.runs import PROJECT_ROOT


def _data(n=30, d=4, seed=3):
    return np.random.default_rng(seed).normal(size=(n, d))


def test_bmu_matches_brute_force_euclidean():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    W = rng.normal(size=(12, 5))
    bmu, dist = matching_units(X, W)
    for i, x in enumerate(X):
        norms = [np.linalg.norm(x - w) for w in W]
        order = np.argsort(norms, kind="stable")
        assert bmu[i] == order[0] == best_matching_unit(W, x)
        assert dist[i] == pytest.approx(norms[order[0]], abs=1e-12)


def test_bmu_tie_goes_to_lowest_index():
    W = np.array([[5.0, 5.0], [1.0, 1.0], [9.0, 9.0], [1.0, 1.0]])
    x = np.array([[1.2, 0.9]])
    assert best_matching_unit(W, x[0]) == 1
    bmu, _ = matching_units(x, W)
    assert bmu[0] == 1


def test_identical_vectors_tie_exactly():
    W = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    d2 = squared_distances(np.array([[0.7, -0.4, 0.25]]), W)
    assert d2[0, 0] == d2[0, 1]


def test_step_includes_the_boundary_and_excludes_beyond():
    one = lambda d, r: float(neighborhood(np.array([d]), r)[0])  # noqa: E731
    for d in (0.0, 0.5, 1.0, 1.0000000000000002):
        assert one(d, 1.0) == 1.0
    assert one(1.01, 1.0) == 0.0
    assert one(0.0, 0.25) == 1.0


def test_step_is_non_increasing_in_distance_and_radius():
    d = np.linspace(0, 4, 41)
    h = neighborhood(d, 1.5)
    assert np.all(np.diff(h) <= 0)
    radii = [3.0, 2.0, 1.0, 0.5, 0.1]
    values = np.array([neighborhood(d, r) for r in radii])
    assert np.all(np.diff(values, axis=0) <= 0)


def test_winner_always_has_full_influence_at_any_positive_radius():
    for r in (1e-6, 0.1, 1.0, 50.0):
        assert neighborhood(np.array([0.0]), r)[0] == 1.0


def test_neighborhood_rejects_bad_radius():
    for r in (0.0, -1.0):
        with pytest.raises(ValueError):
            neighborhood(np.array([0.0, 1.0]), r)


@pytest.mark.parametrize("kind", ["inverse"])
def test_inverse_decay_is_start_over_iteration_and_never_increases(kind):
    total = 50
    values = [decay(0.5, s, kind) for s in range(total)]
    assert values[0] == 0.5
    assert values[-1] == pytest.approx(0.5 / total)
    assert all(b <= a + 1e-15 for a, b in zip(values, values[1:]))
    assert decay(0.5, 0, kind) == 0.5


def test_decay_constant_and_errors():
    assert all(decay(0.3, s, "constant") == 0.3 for s in range(20))
    with pytest.raises(ValueError):
        decay(1.0, 3, "cosine")


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


def _cfg(**kw):
    base = dict(rows=3, cols=3, epochs=6, eta0=0.5, radius0=2.0)
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
    assert res.history["eta"][-1] == pytest.approx(0.5 / (8 * len(_data())))


def test_constant_radius():
    res = train_som(_data(), _cfg(radius0=2.0, radius_decay="constant"), np.random.default_rng(0))
    assert np.all(res.history["radius"] == 2.0)
    res = train_som(_data(), _cfg(radius_decay="constant"), np.random.default_rng(0))
    assert np.all(res.history["radius"] == 2.0)


@pytest.mark.parametrize("init", ["sample", "random"])
def test_weights_stay_in_the_convex_hull_of_the_data(init):
    X = _data()
    res = train_som(X, _cfg(init=init, eta0=0.9), np.random.default_rng(4))
    assert np.all(res.weights >= X.min(axis=0) - 1e-9)
    assert np.all(res.weights <= X.max(axis=0) + 1e-9)


def test_one_by_n_grid_trains():
    res = train_som(_data(), _cfg(rows=1, cols=6, radius0=2.0), np.random.default_rng(0))
    assert res.weights.shape == (6, 4)


def test_train_rejects_unknown_options():
    with pytest.raises(ValueError):
        train_som(_data(), _cfg(topology="tri"), np.random.default_rng(0))


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
