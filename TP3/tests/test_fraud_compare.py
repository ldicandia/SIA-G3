import pytest
from scripts.fraud_compare import (
    split_predictions,
    mse,
    r2,
    clip_fraction,
    saturation_fraction,
    select_model
)

def test_split_predictions():
    predictions = [
        {"expected": 1.0, "predicted": 0.9},
        {"expected": 0.0, "predicted": 0.1},
        {"expected": 1.0, "predicted": 0.8},
        {"expected": 0.0, "predicted": 0.2},
    ]
    labels_rows = [
        {"row_id": "0", "split": "train"},
        {"row_id": "1", "split": "test"},
        {"row_id": "2", "split": "train"},
        {"row_id": "3", "split": "test"},
    ]
    train_pairs, test_pairs = split_predictions(predictions, labels_rows)
    assert len(train_pairs) == 2
    assert len(test_pairs) == 2
    assert train_pairs == [(1.0, 0.9), (1.0, 0.8)]
    assert test_pairs == [(0.0, 0.1), (0.0, 0.2)]

def test_mse():
    pairs = [(1.0, 1.0), (0.0, 1.0)]
    assert mse(pairs) == 0.5

def test_clip_fraction():
    predictions = [
        {"predicted": 0.5},
        {"predicted": 1.2},
        {"predicted": -0.1},
        {"predicted": 0.9}
    ]
    assert clip_fraction(predictions, low=0.0, high=1.0) == 0.5

def test_saturation_fraction():
    predictions = [
        {"predicted": 0.01},
        {"predicted": 0.5},
        {"predicted": 0.99},
        {"predicted": 0.5}
    ]
    assert saturation_fraction(predictions, epsilon=0.02) == 0.5

def test_select_model():
    # Sigmoid within tie threshold -> chooses sigmoid
    id_metrics = {"test_r2": 0.80}
    sig_metrics = {"test_r2": 0.805}
    assert select_model(id_metrics, sig_metrics, r2_tie_threshold=0.01) == "sigmoid"

    # Identity clearly superior by > 0.01
    id_metrics_higher = {"test_r2": 0.90}
    sig_metrics_lower = {"test_r2": 0.85}
    assert select_model(id_metrics_higher, sig_metrics_lower, r2_tie_threshold=0.01) == "identity"

    # Sigmoid clearly superior by > 0.01
    id_metrics_lower = {"test_r2": 0.70}
    sig_metrics_higher = {"test_r2": 0.85}
    assert select_model(id_metrics_lower, sig_metrics_higher, r2_tie_threshold=0.01) == "sigmoid"
