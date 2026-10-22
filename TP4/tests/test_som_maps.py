"""Map readings of the trained SOM: U-matrix, neighbor distances, prototypes and PC planes."""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
import pytest

from tp4.analysis.som_maps import edge_feature_diffs, u_edges, u_matrix
from tp4.cli import main
from tp4.models.som import make_grid
from tp4.runs import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "datos" / "creadores.csv"


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _config() -> dict:
    return json.loads((PROJECT_ROOT / "configs" / "som.json").read_text(encoding="utf-8"))


def _grid_from_config():
    p = _config()["params"]
    rows = p["rows"]
    cols = p["cols"] if p["cols"] is not None else rows
    return make_grid(rows, cols, p["topology"])


@pytest.fixture(scope="module")
def som_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("som_maps_run") / "runs"
    config = _config()
    config["params"]["n_repeats"] = 2
    path = out.parent / "som.json"
    path.write_text(json.dumps(config))
    assert main(["run", str(path), "--out-root", str(out)]) == 0
    return out / "som"


def test_u_matrix_2x2_hand_computed():
    grid = make_grid(2, 2, "rect")
    W = np.array([[0.0], [1.0], [2.0], [4.0]])
    assert u_matrix(W, grid) == pytest.approx([1.5, 2.0, 2.0, 2.5])


def test_u_edges_2x2_ordered_by_distance_then_indices():
    grid = make_grid(2, 2, "rect")
    W = np.array([[0.0], [1.0], [2.0], [4.0]])
    edges = u_edges(W, grid)
    assert list(edges.columns) == ["neuron_a", "neuron_b", "distance"]
    got = list(zip(edges["neuron_a"], edges["neuron_b"], edges["distance"]))
    assert got == [(1, 3, 3.0), (0, 2, 2.0), (2, 3, 2.0), (0, 1, 1.0)]


def test_u_matrix_1x3_hand_computed():
    grid = make_grid(1, 3, "rect")
    W = np.array([[0.0], [1.0], [3.0]])
    assert u_matrix(W, grid) == pytest.approx([1.0, 1.5, 2.0])


def test_u_matrix_uses_euclidean_norm_not_per_coordinate_mean():
    grid = make_grid(1, 2, "rect")
    W = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert u_matrix(W, grid) == pytest.approx([5.0, 5.0])


def test_hex_interior_neuron_averages_over_exactly_six_neighbors():
    grid = make_grid(3, 3, "hex")
    assert grid.neighbors(4).size == 6
    W = np.zeros((9, 1))
    neighbors = grid.neighbors(4)
    W[neighbors, 0] = np.arange(1.0, 7.0)  # distances 1..6 from the (zero) center
    assert u_matrix(W, grid)[4] == pytest.approx(np.mean(np.arange(1.0, 7.0)))


def test_edge_feature_diffs_orders_by_magnitude_with_sign_and_name_ties():
    W = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, -3.0, 3.0, 0.5]])
    text = edge_feature_diffs(W, ["a", "b", "c", "d"], 0, 1)
    assert text == "b:-3.00;c:+3.00;a:+1.00"  # |-3| ties |3|: by name
    assert edge_feature_diffs(W, ["a", "b", "c", "d"], 1, 0, n=1) == "b:+3.00"


def test_cli_u_matrix_artifacts(som_run):
    grid = _grid_from_config()
    edges = _read(som_run / "u_edges.csv")
    assert len(edges) == len(grid.adjacency_pairs())
    assert list(edges[0]) == [
        "neuron_a", "row_a", "col_a", "neuron_b", "row_b", "col_b", "distance", "creators_a", "creators_b", "top_diff",
    ]
    distances = [float(r["distance"]) for r in edges]
    assert distances == sorted(distances, reverse=True)
    neurons = _read(som_run / "neurons.csv")
    assert "u_distance" in neurons[0]
    assert (som_run / "u_matrix.png").stat().st_size > 0
    summary = json.loads((som_run / "summary.json").read_text(encoding="utf-8"))
    for key in ("u_mean", "u_max", "u_max_edge"):
        assert key in summary
    top = edges[0]
    assert summary["u_max_edge"] == f"{top['neuron_a']}~{top['neuron_b']}"


def test_neurons_u_distance_matches_the_edge_table(som_run):
    neurons = _read(som_run / "neurons.csv")
    edges = _read(som_run / "u_edges.csv")
    grid = _grid_from_config()
    dist = {(int(r["neuron_a"]), int(r["neuron_b"])): float(r["distance"]) for r in edges}
    for i, row in enumerate(neurons):
        near = grid.neighbors(i)
        expected = np.mean([dist[(min(i, int(j)), max(i, int(j)))] for j in near])
        assert float(row["u_distance"]) == pytest.approx(expected, rel=1e-8)


from tp4.analysis.pca import fit_pca, standardize  # noqa: E402
from tp4.analysis.som_maps import pc_projection, prototypes_original_units  # noqa: E402
from tp4.features import load_design  # noqa: E402


def test_prototypes_original_units_zero_and_one_weights():
    mean = np.array([10.0, -2.0, 5.0])
    std = np.array([2.0, 0.5, 4.0])
    features = ["a", "b", "c"]
    zero = prototypes_original_units(np.zeros((4, 3)), mean, std, features)
    one = prototypes_original_units(np.ones((4, 3)), mean, std, features)
    assert list(zero.columns) == features
    assert zero.to_numpy() == pytest.approx(np.tile(mean, (4, 1)))
    assert one.to_numpy() == pytest.approx(np.tile(mean + std, (4, 1)))


def test_pc_projection_is_weights_times_components_transposed():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(7, 5))
    components = rng.normal(size=(4, 5))
    assert pc_projection(W, components, 2) == pytest.approx(W @ components[:2].T, abs=1e-12)
    assert pc_projection(W, components).shape == (7, 2)


def test_pc_projection_of_the_creator_rows_equals_pca_scores():
    design = load_design(DATA_PATH, PROJECT_ROOT / "configs" / "feature_set.json")
    Z, _mean, _std = standardize(design.X)
    pca = fit_pca(Z, list(design.X.columns))
    assert pc_projection(Z, pca.components, 2) == pytest.approx(pca.scores[:, :2], abs=1e-10)


def test_cli_prototypes_planes_and_pc_columns(som_run):
    design = load_design(DATA_PATH, PROJECT_ROOT / "configs" / "feature_set.json")
    features = list(design.X.columns)
    grid = _grid_from_config()
    protos = _read(som_run / "prototypes.csv")
    assert list(protos[0]) == ["neuron", "row", "col", *features]
    assert len(protos[0]) == len(features) + 3 and len(protos) == grid.n_neurons
    for name in ("component_planes.png", "pc_planes.png"):
        assert (som_run / name).stat().st_size > 0
    for table in ("neurons.csv", "bmus.csv"):
        header = list(_read(som_run / table)[0])
        assert "pc1" in header and "pc2" in header, table


def test_prototype_values_are_weights_in_original_units(som_run):
    design = load_design(DATA_PATH, PROJECT_ROOT / "configs" / "feature_set.json")
    features = list(design.X.columns)
    Z, mean, std = standardize(design.X)
    weights = _read(som_run / "weights.csv")
    protos = _read(som_run / "prototypes.csv")
    w = np.array([[float(r[f]) for f in features] for r in weights])
    p = np.array([[float(r[f]) for f in features] for r in protos])
    assert p == pytest.approx(mean + w * std, rel=1e-6, abs=1e-6)


def test_pc_columns_and_summary_agree_with_pca(som_run):
    design = load_design(DATA_PATH, PROJECT_ROOT / "configs" / "feature_set.json")
    Z, _mean, _std = standardize(design.X)
    pca = fit_pca(Z, list(design.X.columns))
    bmus = _read(som_run / "bmus.csv")
    scores = np.array([[float(r["pc1"]), float(r["pc2"])] for r in bmus])
    assert scores == pytest.approx(pca.scores[:, :2], abs=1e-6)
    summary = json.loads((som_run / "summary.json").read_text(encoding="utf-8"))
    assert summary["pc1_explained_variance_pct"] == round(float(pca.explained_variance_ratio[0]) * 100, 1)
    assert summary["pc2_explained_variance_pct"] == round(float(pca.explained_variance_ratio[1]) * 100, 1)


def test_neuron_pc_columns_are_the_prototype_projection(som_run):
    design = load_design(DATA_PATH, PROJECT_ROOT / "configs" / "feature_set.json")
    features = list(design.X.columns)
    Z, _mean, _std = standardize(design.X)
    pca = fit_pca(Z, features)
    weights = _read(som_run / "weights.csv")
    w = np.array([[float(r[f]) for f in features] for r in weights])
    neurons = _read(som_run / "neurons.csv")
    got = np.array([[float(r["pc1"]), float(r["pc2"])] for r in neurons])
    assert got == pytest.approx(pc_projection(w, pca.components, 2), abs=1e-6)
