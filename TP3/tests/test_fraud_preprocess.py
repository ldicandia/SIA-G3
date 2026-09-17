import pytest
import math
from pathlib import Path
from scripts.fraud_preprocess import (
    split_indices,
    standardize_fit,
    standardize_apply,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

def test_split_indices():
    n = 100
    seed = 42
    test_fraction = 0.2
    train_idx, test_idx = split_indices(n, seed, test_fraction)

    assert len(train_idx) == 80
    assert len(test_idx) == 20
    assert sorted(train_idx + test_idx) == list(range(n))
    assert train_idx == sorted(train_idx)
    assert test_idx == sorted(test_idx)

    # Calling twice returns identical lists
    t1, te1 = split_indices(n, seed, test_fraction)
    assert t1 == train_idx
    assert te1 == test_idx

def test_standardize_fit_and_apply():
    columns = ["c1", "c2"]
    rows = [
        {"c1": "10.0", "c2": "5.0"},
        {"c1": "20.0", "c2": "5.0"},
        {"c1": "30.0", "c2": "5.0"},
    ]
    stats = standardize_fit(rows, columns)
    assert stats["c1"]["mean"] == 20.0
    # Sample or population std: std = sqrt(((10-20)^2 + (20-20)^2 + (30-20)^2)/3) or /2
    # Check zero variance guard:
    assert stats["c2"]["std"] == 1.0

    # Applying to same rows yields mean ~ 0 for non-zero variance
    applied_c1 = [standardize_apply(r, stats, columns)["c1"] for r in rows]
    assert abs(sum(applied_c1) / len(applied_c1)) < 1e-9

def test_headers_never_contain_flagged_fraud(tmp_path):
    from scripts.fraud_preprocess import write_feature_csv
    rows = [{"f1": "1.0", "big_model_fraud_probability": "0.5", "flagged_fraud": "1"}]
    stats = {"f1": {"mean": 0.0, "std": 1.0}}
    out_file = tmp_path / "test.csv"
    write_feature_csv(out_file, rows, stats, feature_cols=["f1"], target_col="big_model_fraud_probability")
    
    content = out_file.read_text()
    first_line = content.splitlines()[0]
    assert "flagged_fraud" not in first_line
    assert first_line == "f1,big_model_fraud_probability"
