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


# --------------------------------------------------------------------------- Task 2: planes in original units and PC planes

from tp4.analysis.pca import fit_pca, standardize  # noqa: E402
from tp4.analysis.som_maps import pc_projection, prototypes_original_units, reconstruct_topic_art  # noqa: E402
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


def test_reconstruct_topic_art_is_100_minus_the_other_three():
    frame = pd.DataFrame(
        {"topic_football_pct": [30.0], "topic_makeup_pct": [20.0], "topic_dogs_pct": [10.0]}
    )
    out = reconstruct_topic_art(frame)
    assert out.tolist() == [40.0]


@pytest.mark.parametrize("missing", ["topic_football_pct", "topic_makeup_pct", "topic_dogs_pct"])
def test_reconstruct_topic_art_names_the_missing_column(missing):
    frame = pd.DataFrame({"topic_football_pct": [30.0], "topic_makeup_pct": [20.0], "topic_dogs_pct": [10.0]})
    with pytest.raises(ValueError, match=missing):
        reconstruct_topic_art(frame.drop(columns=[missing]))


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
    assert list(protos[0]) == ["neuron", "row", "col", *features, "topic_art_pct_rec"]
    assert len(protos[0]) == 29 and len(protos) == grid.n_neurons
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
    rec = np.array([float(r["topic_art_pct_rec"]) for r in protos])
    cols = {c: np.array([float(r[c]) for r in protos]) for c in features if c.startswith("topic_")}
    assert rec == pytest.approx(100.0 - cols["topic_football_pct"] - cols["topic_makeup_pct"] - cols["topic_dogs_pct"], abs=1e-6)


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


# --------------------------------------------------------------------------- Task 3: regions, profiles, eta^2

from collections import deque  # noqa: E402

from tp4.analysis.som_maps import (  # noqa: E402
    MIN_REGION_CREATORS,
    REGION_K_RANGE,
    choose_n_regions,
    correlation_ratio,
    region_border_ratio,
    region_hierarchy,
    region_profiles,
    region_scan,
    segment_regions,
)
from tp4.features import TOPIC_COLUMNS  # noqa: E402


def _connected(grid, members):
    members = set(int(m) for m in members)
    start = next(iter(members))
    seen, queue = {start}, deque([start])
    while queue:
        cur = queue.popleft()
        for nb in grid.neighbors(cur):
            nb = int(nb)
            if nb in members and nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return seen == members


def test_region_hierarchy_splits_the_1x6_line_at_the_gap():
    grid = make_grid(1, 6, "rect")
    W = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    labels = region_hierarchy(W, grid, [2])[2]
    assert labels.tolist() == [1, 1, 1, 2, 2, 2]
    assert region_border_ratio(W, grid, labels) > 1.0
    assert segment_regions(W, grid, 2).tolist() == [1, 1, 1, 2, 2, 2]


@pytest.mark.parametrize("k", range(2, 9))
def test_regions_are_k_connected_numbered_by_smallest_neuron_and_deterministic(k):
    grid = make_grid(5, 5, "hex")
    W = np.random.default_rng(7).normal(size=(25, 4))
    labels = segment_regions(W, grid, k)
    assert sorted(set(labels.tolist())) == list(range(1, k + 1))
    for r in range(1, k + 1):
        assert _connected(grid, np.flatnonzero(labels == r))
    firsts = [int(np.flatnonzero(labels == r)[0]) for r in range(1, k + 1)]
    assert firsts == sorted(firsts)
    assert np.array_equal(labels, segment_regions(W, grid, k))


def test_hierarchy_snapshots_match_single_k_calls():
    grid = make_grid(4, 4, "rect")
    W = np.random.default_rng(3).normal(size=(16, 3))
    snaps = region_hierarchy(W, grid, [2, 3, 5, 8])
    for k, labels in snaps.items():
        assert np.array_equal(labels, segment_regions(W, grid, k))


def test_extreme_k_values():
    grid = make_grid(3, 3, "hex")
    W = np.random.default_rng(1).normal(size=(9, 2))
    assert segment_regions(W, grid, 9).tolist() == list(range(1, 10))
    assert segment_regions(W, grid, 1).tolist() == [1] * 9
    with pytest.raises(ValueError):
        segment_regions(W, grid, 10)


def test_region_border_ratio_nan_when_a_side_is_empty():
    grid = make_grid(1, 3, "rect")
    W = np.array([[0.0], [1.0], [3.0]])
    assert np.isnan(region_border_ratio(W, grid, np.array([1, 1, 1])))
    assert np.isnan(region_border_ratio(W, grid, np.array([1, 2, 3])))


def test_correlation_ratio_cases():
    assert correlation_ratio(np.array([0.0, 0.0, 10.0, 10.0]), np.array([1, 1, 2, 2])) == pytest.approx(1.0)
    assert correlation_ratio(np.array([3.0, 3.0, 3.0]), np.array([1, 2, 2])) == 0.0
    assert correlation_ratio(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1])) == 0.0


def test_choose_n_regions_picks_best_qualifying_ratio_ties_to_smaller_k():
    scan = pd.DataFrame(
        {
            "k": [2, 3, 4, 5],
            "u_border_ratio": [1.2, 1.5, 1.5, 2.0],
            "min_creators": [5, 4, 3, 2],
            "n_regions_without_creators": [0, 0, 0, 0],
        }
    )
    assert MIN_REGION_CREATORS == 3 and REGION_K_RANGE == (2, 8)
    assert choose_n_regions(scan) == 3  # k=5 has the best ratio but a region with 2 creators; 3 and 4 tie
    assert choose_n_regions(scan, min_creators=1) == 5


def test_choose_n_regions_returns_2_when_none_qualifies():
    scan = pd.DataFrame(
        {"k": [3, 4], "u_border_ratio": [1.5, 2.0], "min_creators": [0, 2], "n_regions_without_creators": [1, 0]}
    )
    assert choose_n_regions(scan) == 2
    assert choose_n_regions(scan.iloc[0:0]) == 2


def test_region_scan_one_row_per_k():
    grid = make_grid(1, 6, "rect")
    W = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    bmu = np.array([0, 0, 1, 3, 4, 5, 5])
    scan = region_scan(W, grid, bmu, range(2, 6))
    assert list(scan.columns) == ["k", "u_border_ratio", "min_creators", "n_regions_without_creators"]
    assert scan["k"].tolist() == [2, 3, 4, 5]
    row2 = scan.iloc[0]
    assert row2["min_creators"] == 3 and row2["n_regions_without_creators"] == 0 and row2["u_border_ratio"] > 1


def test_region_profiles_empty_region_has_zero_creators_and_nan_means():
    Z = pd.DataFrame({"a": [-1.0, 1.0, 0.0], "b": [0.5, -0.5, 0.0]})
    raw = pd.DataFrame({"a": [1.0, 3.0, 2.0], "b": [10.0, 0.0, 5.0], "topic_art_pct": [0.0, 30.0, 0.0]})
    prof = region_profiles(Z, raw, np.array([1, 1, 3]), 3)
    assert list(prof.columns) == ["region", "feature", "mean_z", "mean_raw", "n_creators"]
    assert len(prof) == 3 * 3  # regions x (a, b, topic_art_pct)
    empty = prof[prof["region"] == 2]
    assert (empty["n_creators"] == 0).all() and empty["mean_z"].isna().all() and empty["mean_raw"].isna().all()
    r1 = prof[prof["region"] == 1].set_index("feature")
    assert r1.loc["a", "mean_z"] == pytest.approx(0.0) and r1.loc["a", "mean_raw"] == pytest.approx(2.0)
    assert r1.loc["b", "mean_raw"] == pytest.approx(5.0)
    art_z = (raw["topic_art_pct"] - raw["topic_art_pct"].mean()) / raw["topic_art_pct"].std(ddof=0)
    assert r1.loc["topic_art_pct", "mean_z"] == pytest.approx(art_z.iloc[:2].mean())
    assert r1.loc["topic_art_pct", "mean_raw"] == pytest.approx(15.0)
    csv_text = prof.to_csv(index=False, float_format="%.10g")
    assert "nan" not in csv_text.lower()


def test_draw_region_borders_draws_one_segment_per_pair():
    from tp4.plotting import draw_grid_cells, draw_region_borders, plt

    grid = make_grid(2, 2, "rect")
    fig, ax = plt.subplots()
    draw_grid_cells(ax, grid.positions, "rect", np.zeros(4))
    before = len(ax.collections)
    draw_region_borders(ax, grid.positions, "rect", [(0, 1), (2, 3)])
    assert len(ax.collections) == before + 1
    segments = ax.collections[-1].get_segments()
    assert len(segments) == 2
    seg = np.asarray(segments[0])  # border between (0,0) and (0,1): vertical unit segment at x=0.5
    assert seg[:, 0] == pytest.approx([0.5, 0.5])
    assert abs(seg[1, 1] - seg[0, 1]) == pytest.approx(1.0)
    plt.close(fig)


def test_cli_regions_artifacts(som_run):
    n_creators = len(_read(DATA_PATH))
    grid = _grid_from_config()
    regions = _read(som_run / "regions.csv")
    assert list(regions[0]) == [
        "region", "n_neurons", "n_creators", "creators", "topic_counts", "top_high", "top_low", "pc1_mean", "pc2_mean",
    ]
    summary = json.loads((som_run / "summary.json").read_text(encoding="utf-8"))
    for key in ("n_regions", "n_regions_auto", "region_creator_counts", "u_border_ratio", "pc1_region_eta2", "pc2_region_eta2"):
        assert key in summary, key
    assert len(regions) == summary["n_regions"]
    assert sum(int(r["n_creators"]) for r in regions) == n_creators == sum(summary["region_creator_counts"])
    assert sum(int(r["n_neurons"]) for r in regions) == grid.n_neurons
    listed = [name for r in regions for name in (r["creators"].split(";") if r["creators"] else [])]
    assert sorted(listed) == sorted(r["name"] for r in _read(DATA_PATH))
    assert 0.0 <= summary["pc1_region_eta2"] <= 1.0 and 0.0 <= summary["pc2_region_eta2"] <= 1.0
    for name in ("region_profiles.csv", "region_scan.csv", "region_map.png", "region_profiles.png", "biplot_regions.png"):
        assert (som_run / name).stat().st_size > 0, name
    for table in ("bmus.csv", "neurons.csv"):
        assert "region" in _read(som_run / table)[0], table
    scan = _read(som_run / "region_scan.csv")
    assert [int(r["k"]) for r in scan] == list(range(2, min(8, grid.n_neurons - 1) + 1))


def test_cli_regions_are_contiguous_and_consistent_with_tables(som_run):
    grid = _grid_from_config()
    neurons = _read(som_run / "neurons.csv")
    labels = np.array([int(r["region"]) for r in neurons])
    for r in sorted(set(labels.tolist())):
        assert _connected(grid, np.flatnonzero(labels == r))
    for row in _read(som_run / "bmus.csv"):
        assert int(row["region"]) == labels[int(row["bmu"])]
    for r in _read(som_run / "regions.csv"):
        assert int(r["n_neurons"]) == int((labels == int(r["region"])).sum())
        counts = dict(part.split(":") for part in r["topic_counts"].split(";"))
        assert list(counts) == [c.removeprefix("topic_").removesuffix("_pct") for c in TOPIC_COLUMNS]
        assert sum(int(v) for v in counts.values()) == int(r["n_creators"])


def test_cli_profiles_cover_every_region_and_feature_without_nan_text(som_run):
    summary = json.loads((som_run / "summary.json").read_text(encoding="utf-8"))
    prof = _read(som_run / "region_profiles.csv")
    assert len(prof) == summary["n_regions"] * 26
    assert "nan" not in (som_run / "region_profiles.csv").read_text(encoding="utf-8").lower()
    assert "nan" not in (som_run / "regions.csv").read_text(encoding="utf-8").lower()


def test_cli_auto_region_count_follows_the_scan(som_run):
    scan = pd.read_csv(som_run / "region_scan.csv")
    summary = json.loads((som_run / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_regions_auto"] is True
    assert summary["n_regions"] == choose_n_regions(scan)


def test_explicit_n_regions_with_an_empty_region_leaves_blank_cells(tmp_path, write_config):
    cfg = write_config(base="configs/som.json", plots=False, **{"params.n_regions": 40})
    assert main(["run", str(cfg), "--out-root", str(tmp_path / "out")]) == 0
    run_dir = tmp_path / "out" / "som"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_regions"] == 40 and summary["n_regions_auto"] is False
    regions = _read(run_dir / "regions.csv")
    assert len(regions) == 40
    empty = [r for r in regions if int(r["n_creators"]) == 0]
    assert empty, "40 regions on 64 neurons with 32 empty neurons must include a creator-free region"
    for r in empty:
        assert r["creators"] == "" and r["top_high"] == "" and r["top_low"] == "" and r["pc1_mean"] == ""
    assert "nan" not in (run_dir / "regions.csv").read_text(encoding="utf-8").lower()
    assert sum(int(r["n_creators"]) for r in regions) == len(_read(DATA_PATH))


@pytest.mark.parametrize("bad", [0, 65, -1, 2.5, True, "3"])
def test_invalid_n_regions_exits_2(tmp_path, write_config, capsys, bad):
    cfg = write_config(base="configs/som.json", plots=False, **{"params.n_regions": bad})
    out = tmp_path / "out"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not (out / "som").exists()
