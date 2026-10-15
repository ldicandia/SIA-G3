from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
import pytest

from tp4.analysis.pca import (
    components_for_threshold,
    fit_pca,
    orient_components,
    standardize,
)
from tp4.cli import main
from tp4.errors import DataError



@pytest.fixture(scope="module")
def pca_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("pca_run") / "runs"
    assert main(["run", "configs/pca.json", "--out-root", str(out)]) == 0
    return out / "pca"


def test_cli_end_to_end_writes_explained_variance_and_scree(pca_run):
    for name in ("explained_variance.csv", "scree.png", "summary.json", "run.json"):
        assert (pca_run / name).stat().st_size > 0
    with (pca_run / "explained_variance.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((pca_run / "summary.json").read_text(encoding="utf-8"))
    assert len(rows) == summary["n_features"]
    ratios = np.array([float(r["explained_variance_ratio"]) for r in rows])
    cum = np.array([float(r["cumulative_ratio"]) for r in rows])
    assert abs(ratios.sum() - 1.0) < 1e-9
    assert np.all(np.diff(cum) >= -1e-12)
    assert abs(cum[-1] - 1.0) < 1e-9
    assert list(rows[0]) == ["component", "eigenvalue", "explained_variance_ratio", "cumulative_ratio"]


def test_real_data_summary_matches_planning_facts(pca_run):
    summary = json.loads((pca_run / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_features"] == 25
    assert summary["n_samples"] == 50
    assert summary["pc1_explained_variance_pct"] == 53.2
    assert summary["pc2_explained_variance_pct"] == 8.0
    assert summary["k_cum_80"] == 6
    assert summary["k_cum_90"] == 9
    assert summary["kaiser_k"] == 6


def test_eigenvalues_match_correlation_matrix(pca_run):
    rows = pd.read_csv(pca_run / "explained_variance.csv")
    assert rows["eigenvalue"].sum() == pytest.approx(25.0, abs=1e-6)  # trace of a 25x25 correlation matrix


def test_orientation_rule_holds_on_fit_pca():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=list("abcde"))
    Z = standardize(X)[0]
    result = fit_pca(Z, list(X.columns))
    for row in result.components:
        assert row[int(np.argmax(np.abs(row)))] > 0
    np.testing.assert_allclose(result.scores, Z @ result.components.T, atol=1e-10)


def test_orient_components_flips_scores_and_breaks_ties_low():
    comps = np.array([[-0.6, 0.6, 0.5], [0.1, -0.9, 0.2]])
    scores = np.array([[1.0, 2.0], [3.0, 4.0]])
    oriented, flipped = orient_components(comps, scores)
    # first row: tie between |-0.6| and |0.6| goes to index 0, which is negative -> flip
    np.testing.assert_allclose(oriented[0], [0.6, -0.6, -0.5])
    np.testing.assert_allclose(oriented[1], [-0.1, 0.9, -0.2])
    np.testing.assert_allclose(flipped, -scores)


def test_standardize_raises_on_constant_and_nan_columns():
    X = pd.DataFrame({"ok": [1.0, 2.0, 3.0], "flat": [5.0, 5.0, 5.0]})
    with pytest.raises(DataError, match="flat"):
        standardize(X)
    Y = pd.DataFrame({"ok": [1.0, 2.0, 3.0], "holes": [1.0, np.nan, 3.0]})
    with pytest.raises(DataError, match="holes"):
        standardize(Y)


def test_standardize_uses_population_std():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    Z, mean, std = standardize(X)
    assert std[0] == pytest.approx(np.std([1, 2, 3, 4]))
    assert Z.std(axis=0, ddof=0)[0] == pytest.approx(1.0)
    assert mean[0] == pytest.approx(2.5)


def test_components_for_threshold_exact_equality_edge():
    cum = np.array([0.5, 0.79, 0.8, 1.0])
    assert components_for_threshold(cum, 0.8) == 3
    assert components_for_threshold(cum, 0.81) == 4
    assert components_for_threshold(np.array([0.5, 0.6]), 0.99) == 2


def test_two_runs_give_identical_explained_variance(tmp_path, write_config):
    cfg = write_config("configs/pca.json", plots=False)
    a, b = tmp_path / "a", tmp_path / "b"
    assert main(["run", str(cfg), "--out-root", str(a)]) == 0
    assert main(["run", str(cfg), "--out-root", str(b)]) == 0
    assert (a / "pca" / "explained_variance.csv").read_bytes() == (b / "pca" / "explained_variance.csv").read_bytes()


@pytest.mark.parametrize(
    "override",
    [
        {"params.n_components": 0},
        {"params.variance_thresholds": []},
        {"params.variance_thresholds": [1.5]},
        {"params.loadings_pcs": 0},
        {"params.top_n": 0},
        {"params.biplot_max_arrows": 0},
        {"feature_set": ""},
    ],
)
def test_invalid_pca_configs_exit_2(tmp_path, write_config, override, capsys):
    cfg = write_config("configs/pca.json", plots=False, **override)
    assert main(["run", str(cfg), "--out-root", str(tmp_path / "runs")]) == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "pca").exists()


def test_n_components_above_rank_is_a_config_error(tmp_path, write_config, capsys):
    cfg = write_config("configs/pca.json", plots=False, **{"params.n_components": 99})
    assert main(["run", str(cfg), "--out-root", str(tmp_path / "runs")]) == 2
    assert "error:" in capsys.readouterr().err
