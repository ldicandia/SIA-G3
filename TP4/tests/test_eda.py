from __future__ import annotations

import csv
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from tp4.analysis.eda import (
    correlation_matrix,
    describe_columns,
    iqr_outliers,
    tukey_fences,
    zscores,
)
from tp4.cli import main
from tp4.data import load_creators, numeric_columns
from tp4.runs import PROJECT_ROOT

DATA = PROJECT_ROOT / "datos" / "creadores.csv"
OUTLIER_HEADER = "column,creator_id,name,value,lower_fence,upper_fence,side"


def frame(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    data = {"creator_id": [f"C{i:03d}" for i in range(n)], "name": [f"n{i}" for i in range(n)]}
    data.update(cols)
    return pd.DataFrame(data)


def write_csv_file(path, rows, header=("creator_id", "name", "a", "b")):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def tiny_feature_set(tmp_path):
    """A feature set for the 2-column synthetic CSVs: drops `b`, keeps `a`."""
    path = tmp_path / "tiny_fs.json"
    path.write_text(json.dumps({"drop": {"b": "columna de prueba"}}), encoding="utf-8")
    return str(path)


def test_tukey_fences_linear_interpolation():
    lo, hi = tukey_fences(pd.Series([1, 2, 3, 4, 5]), 1.5)
    assert (lo, hi) == (-1.0, 7.0)


def test_value_exactly_on_upper_fence_is_not_an_outlier():
    # Q1=2.25, Q3=4.75, IQR=2.5 -> upper fence 8.5 (exact in binary)
    on = frame(x=[1, 2, 3, 4, 5, 8.5])
    assert tukey_fences(on["x"], 1.5)[1] == 8.5
    assert iqr_outliers(on, ["x"], 1.5).empty

    above = frame(x=[1, 2, 3, 4, 5, 8.6])
    out = iqr_outliers(above, ["x"], 1.5)
    assert out["value"].tolist() == [8.6]
    assert out["side"].tolist() == ["high"]


def test_value_exactly_on_lower_fence_is_not_an_outlier():
    on = frame(x=[-2.5, 1, 2, 3, 4, 5])
    assert tukey_fences(on["x"], 1.5)[0] == -2.5
    assert iqr_outliers(on, ["x"], 1.5).empty

    below = frame(x=[-2.6, 1, 2, 3, 4, 5])
    out = iqr_outliers(below, ["x"], 1.5)
    assert out["side"].tolist() == ["low"]


def test_constant_column_has_no_outliers_no_warnings_no_division_error():
    df = frame(x=[5.0] * 10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        desc = describe_columns(df, 1.5).set_index("column")
        out = iqr_outliers(df, ["x"], 1.5)
        z = zscores(df, ["x"])
    assert desc.loc["x", "std"] == 0
    assert desc.loc["x", "iqr"] == 0
    assert desc.loc["x", "n_outliers"] == 0
    assert out.empty
    assert (z["x"] == 0.0).all()


def test_nan_is_counted_as_missing_and_never_an_outlier():
    df = frame(x=[1, 2, 3, 4, 5, 100, np.nan])
    desc = describe_columns(df, 1.5).set_index("column")
    assert desc.loc["x", "n_missing"] == 1
    out = iqr_outliers(df, ["x"], 1.5)
    assert out["value"].tolist() == [100.0]
    assert not out["value"].isna().any()


def test_outliers_ordered_by_column_position_then_creator_id():
    # columns passed as ["b", "a"]: b's outliers must come first even though "a" < "b".
    df = frame(
        a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 500, 600, 700],
        b=[1, 2, 3, 4, 5, 6, 7, 8, 9, 800, 900, 1000][::-1],
    )
    out = iqr_outliers(df, ["b", "a"], 1.5)
    assert out["column"].tolist() == sorted(out["column"], key=["b", "a"].index)
    for _, group in out.groupby("column", sort=False):
        assert group["creator_id"].tolist() == sorted(group["creator_id"])


def test_empty_outliers_frame_has_the_exact_columns():
    out = iqr_outliers(frame(x=[1.0, 2.0, 3.0, 4.0]), ["x"], 1.5)
    assert out.empty
    assert list(out.columns) == OUTLIER_HEADER.split(",")


def test_analysis_functions_do_not_modify_the_input():
    df = load_creators(DATA)
    before = df.copy(deep=True)
    cols = numeric_columns(df)
    describe_columns(df, 1.5)
    iqr_outliers(df, cols, 1.5)
    correlation_matrix(df, cols)
    zscores(df, cols)
    pd.testing.assert_frame_equal(df, before)


def test_describe_columns_covers_every_csv_column(csv_rows):
    df = load_creators(DATA)
    desc = describe_columns(df, 1.5)
    assert len(desc) == df.shape[1]
    assert desc["column"].tolist() == list(df.columns)
    ids = desc.set_index("column")
    assert ids.loc["creator_id", "n_unique"] == csv_rows(DATA)
    assert ids.loc["name", "n_unique"] == csv_rows(DATA)
    for col in (
        "n_missing", "n_unique", "mean", "std", "min", "q1", "median", "q3", "max",
        "iqr", "lower_fence", "upper_fence", "n_outliers",
    ):
        assert col in desc.columns
    assert pd.isna(ids.loc["creator_id", "mean"])


def test_correlation_matrix_is_square_ordered_with_unit_diagonal():
    df = load_creators(DATA)
    cols = numeric_columns(df)
    corr = correlation_matrix(df, cols)
    assert corr.shape == (len(cols), len(cols))
    assert list(corr.index) == cols and list(corr.columns) == cols
    assert (np.diag(corr.to_numpy()) == 1.0).all()


def test_zero_row_csv_is_rejected_before_any_run_is_published(tmp_path, out_root, write_config, capsys):
    empty = write_csv_file(tmp_path / "empty.csv", [])
    cfg = write_config(data=str(empty))
    assert main(["run", str(cfg), "--out-root", str(out_root)]) == 2
    assert "zero data rows" in capsys.readouterr().err
    assert not (out_root / "eda").exists()
    assert not list(out_root.glob(".staging-*"))


def test_non_numeric_column_is_rejected_naming_column_and_creator(tmp_path, out_root, write_config, capsys):
    bad = write_csv_file(tmp_path / "bad.csv", [("C1", "x", 1, 2), ("C2", "y", "oops", 3)])
    assert main(["run", str(write_config(data=str(bad))), "--out-root", str(out_root)]) == 2
    err = capsys.readouterr().err
    assert "'a'" in err and "C2" in err


def test_header_only_outliers_csv_and_missing_values_reported(tmp_path, out_root, write_config):
    rows = [("C1", "a1", 1, 10), ("C2", "a2", 2, ""), ("C3", "a3", 3, 30), ("C4", "a4", 4, 40)]
    src = write_csv_file(tmp_path / "small.csv", rows)
    assert main(["run", str(write_config(data=str(src), plots=False, feature_set=tiny_feature_set(tmp_path))), "--out-root", str(out_root)]) == 0
    run_dir = out_root / "eda"
    lines = (run_dir / "outliers.csv").read_text(encoding="utf-8").splitlines()
    assert lines == [OUTLIER_HEADER]
    cols = pd.read_csv(run_dir / "columns.csv").set_index("column")
    assert cols.loc["b", "n_missing"] == 1
    assert cols.loc["a", "n_missing"] == 0


def test_perfectly_correlated_pair_is_named_alphabetically(tmp_path, out_root, write_config):
    rows = [("C1", "a1", 1, 2), ("C2", "a2", 2, 4), ("C3", "a3", 3, 6), ("C4", "a4", 4, 8)]
    src = write_csv_file(tmp_path / "corr.csv", rows)
    main(["run", str(write_config(data=str(src), plots=False, feature_set=tiny_feature_set(tmp_path))), "--out-root", str(out_root)])
    summary = json.loads((out_root / "eda" / "summary.json").read_text(encoding="utf-8"))
    assert summary["max_abs_correlation_pair"] == "a~b"
    assert summary["max_abs_correlation"] == 1.0


def test_real_data_run_writes_all_artifacts_and_keeps_every_creator(out_root, csv_rows):
    assert main(["run", "configs/eda.json", "--out-root", str(out_root)]) == 0
    run_dir = out_root / "eda"
    expected = [
        "run.json", "summary.json", "columns.csv", "outliers.csv", "correlation.csv",
        "distributions.png", "boxplots_standardized.png", "scale_ranges.png",
        "correlation_heatmap.png",
    ]
    for name in expected:
        assert (run_dir / name).stat().st_size > 0, name

    source_ids = set(pd.read_csv(DATA)["creator_id"])
    out = pd.read_csv(run_dir / "outliers.csv")
    assert set(out["creator_id"]) <= source_ids
    assert (run_dir / "outliers.csv").read_text(encoding="utf-8").splitlines()[0] == OUTLIER_HEADER

    columns_lines = (run_dir / "columns.csv").read_text(encoding="utf-8").splitlines()
    assert len(columns_lines) == 1 + pd.read_csv(DATA).shape[1]
    for col in ("n_missing", "n_unique", "q1", "q3", "iqr", "lower_fence", "upper_fence", "n_outliers"):
        assert col in columns_lines[0].split(",")

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for key in ("n_outlier_cells", "n_creators_with_outliers", "max_abs_correlation", "max_abs_correlation_pair"):
        assert key in summary
    # EDA-01 prohibition: outliers are flagged, never dropped.
    assert summary["n_rows"] == csv_rows(DATA)
    assert summary["n_outlier_cells"] == len(out)
    assert summary["n_creators_with_outliers"] == out["creator_id"].nunique()
    assert len(pd.read_csv(run_dir / "correlation.csv")) == summary["n_numeric_columns"]


def test_plots_false_skips_pngs_but_keeps_csvs(out_root, write_config):
    assert main(["run", str(write_config(plots=False)), "--out-root", str(out_root)]) == 0
    run_dir = out_root / "eda"
    assert not list(run_dir.glob("*.png"))
    for name in ("columns.csv", "outliers.csv", "correlation.csv"):
        assert (run_dir / name).is_file()


def test_outlier_k_param_changes_flagged_cells(tmp_path, write_config):
    tight, loose = tmp_path / "tight", tmp_path / "loose"
    main(["run", str(write_config(plots=False, **{"params.outlier_iqr_k": 0.5})), "--out-root", str(tight)])
    main(["run", str(write_config(plots=False, **{"params.outlier_iqr_k": 3.0})), "--out-root", str(loose)])
    n = lambda root: json.loads((root / "eda" / "summary.json").read_text())["n_outlier_cells"]  # noqa: E731
    assert n(tight) > n(loose)
