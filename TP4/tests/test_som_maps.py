"""Map readings of the trained SOM: U-matrix, neighbor walls, planes, regions, profiles."""

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
    assert main(["run", "configs/som.json", "--out-root", str(out)]) == 0
    return out / "som"


# --------------------------------------------------------------------------- Task 1: U-matrix


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
