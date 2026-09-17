import pytest
from scripts.fraud_metrics import (
    confusion_counts,
    precision_recall_f1,
    sweep,
    best_threshold_by_f1
)

def test_confusion_counts():
    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.1, 0.8, 0.2]
    threshold = 0.5
    counts = confusion_counts(y_true, y_prob, threshold)
    assert counts == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}

def test_precision_recall_f1_all_correct():
    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.1, 0.8, 0.2]
    res = precision_recall_f1(y_true, y_prob, 0.5)
    assert res == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

def test_precision_zero_division():
    # No positive predictions -> precision = 0.0 (no exception)
    y_true = [1, 1]
    y_prob = [0.1, 0.2]
    res = precision_recall_f1(y_true, y_prob, 0.5)
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0

def test_recall_zero_division():
    # No actual positives in ground truth -> recall = 0.0 (no exception)
    y_true = [0, 0, 0]
    y_prob = [0.9, 0.8, 0.7]
    res = precision_recall_f1(y_true, y_prob, 0.5)
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0

def test_sweep():
    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.1, 0.8, 0.2]
    thresholds = [0.3, 0.5, 0.7]
    results = sweep(y_true, y_prob, thresholds)
    assert len(results) == 3
    assert [r["threshold"] for r in results] == thresholds

def test_best_threshold_by_f1():
    sweep_results = [
        {"threshold": 0.3, "precision": 0.5, "recall": 1.0, "f1": 0.667},
        {"threshold": 0.5, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        {"threshold": 0.7, "precision": 1.0, "recall": 0.5, "f1": 0.667},
    ]
    best = best_threshold_by_f1(sweep_results)
    assert best["threshold"] == 0.5
    assert best["f1"] == 1.0
