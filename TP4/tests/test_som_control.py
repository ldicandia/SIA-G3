"""som_compare helpers and the `som_control` experiment kind (features map vs 6-PC control map)."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from tp4.analysis.som_compare import adjusted_rand_index, contingency_table, grid_distance_agreement
from tp4.cli import main
from tp4.models.som import make_grid
from tp4.runs import PROJECT_ROOT

CONTROL = "configs/som_pca_control.json"


def test_ari_identical_and_relabeled_partitions_are_one():
    a = [1, 1, 2, 2, 3, 3, 3]
    assert adjusted_rand_index(a, a) == pytest.approx(1.0)
    assert adjusted_rand_index(a, [9, 9, 5, 5, 0, 0, 0]) == pytest.approx(1.0)


def test_ari_known_negative_case():
    assert adjusted_rand_index([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(-0.5)


def test_ari_trivial_partitions():
    assert adjusted_rand_index([1, 1, 1, 1], [7, 7, 7, 7]) == 1.0
    assert adjusted_rand_index([1, 2, 3, 4], [5, 6, 7, 8]) == 1.0  # both all-singletons
    assert adjusted_rand_index([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0  # trivial but different


def test_contingency_table_counts():
    table = contingency_table([2, 2, 1, 1, 1], [5, 6, 5, 5, 6])
    assert list(table.index) == [1, 2]
    assert list(table.columns) == [5, 6]
    assert table.to_numpy().tolist() == [[2, 1], [1, 1]]


def test_grid_distance_agreement_identical_and_constant():
    grid = make_grid(3, 3, "hex")
    bmu = np.array([0, 1, 4, 8, 2, 5])
    assert grid_distance_agreement(grid.distances, bmu, grid.distances, bmu) == pytest.approx(1.0)
    same_neuron = np.zeros(6, dtype=int)
    assert np.isnan(grid_distance_agreement(grid.distances, same_neuron, grid.distances, bmu))


def _summary(path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def paired_runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("control") / "runs"
    assert main(["run", "configs/som.json", "--out-root", str(out)]) == 0
    assert main(["run", CONTROL, "--out-root", str(out)]) == 0
    return out


def test_features_map_reproduces_the_main_som_run(paired_runs):
    som = _summary(paired_runs / "som")
    control = _summary(paired_runs / "som_pca_control")
    assert control["features_quantization_error"] == som["quantization_error"]
    assert control["features_topographic_error"] == som["topographic_error"]
    assert control["features_distance_preservation"] == som["distance_preservation"]
    assert control["n_regions"] == som["n_regions"]
    assert control["pca_components"] == 6
    assert control["pca_variance_kept_pct"] == 83.0
    for key in ("region_ari", "bmu_distance_agreement", "pca_quantization_error", "pca_topographic_error",
                "pca_distance_preservation"):
        assert key in control


def test_control_artifacts_and_creator_rows(paired_runs, csv_rows):
    run = paired_runs / "som_pca_control"
    for name in ("control_comparison.csv", "creator_regions.csv", "region_contingency.csv", "control_region_maps.png"):
        assert (run / name).stat().st_size > 0, name
    with open(run / "creator_regions.csv", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == ["creator_id", "name", "dominant_topic", "bmu_features", "region_features", "bmu_pca", "region_pca"]
    assert csv_rows(run / "creator_regions.csv") == csv_rows(PROJECT_ROOT / "datos" / "creadores.csv")


def _run_expecting_exit_2(cfg, tmp_path, capsys, needle):
    out = tmp_path / "bad_runs"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert needle in err
    assert not (out / "som_pca_control").exists()


def test_seed_must_equal_the_base_seed(tmp_path, capsys, write_config):
    cfg = write_config(CONTROL, seed=7)
    _run_expecting_exit_2(cfg, tmp_path, capsys, "seed")


@pytest.mark.parametrize("k", [0, 26, -1])
def test_pca_components_out_of_range_exits_2(k, tmp_path, capsys, write_config):
    cfg = write_config(CONTROL, **{"params.pca_components": k})
    _run_expecting_exit_2(cfg, tmp_path, capsys, "pca_components")


def _config_with_base(tmp_path, base):
    config = json.loads((PROJECT_ROOT / CONTROL).read_text(encoding="utf-8"))
    config["base"] = base  # write_config has its own `base` argument, so this one is hand-written
    path = tmp_path / "control_bad_base.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_base_must_be_a_som_config(tmp_path, capsys):
    _run_expecting_exit_2(_config_with_base(tmp_path, "configs/pca.json"), tmp_path, capsys, "som")


def test_base_must_be_a_non_empty_string(tmp_path, capsys):
    _run_expecting_exit_2(_config_with_base(tmp_path, ""), tmp_path, capsys, "base")
