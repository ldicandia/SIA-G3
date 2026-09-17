import pytest
from scripts.ejercicio1.fraud_explore import compute_column_stats, count_duplicate_rows, class_balance

def test_compute_column_stats():
    rows = [
        {"val": "1.0", "other": "x"},
        {"val": "2.0", "other": "y"},
        {"val": "3.0", "other": "z"},
    ]
    stats = compute_column_stats(rows, "val")
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    assert stats["mean"] == 2.0
    assert stats["count"] == 3

def test_count_duplicate_rows():
    rows = [
        {"a": "1", "b": "x"},
        {"a": "1", "b": "x"},
        {"a": "2", "b": "y"},
    ]
    assert count_duplicate_rows(rows) == 1

def test_class_balance():
    rows = [
        {"flagged_fraud": "1"},
        {"flagged_fraud": "0"},
        {"flagged_fraud": "1"},
        {"flagged_fraud": "0"},
    ]
    balance = class_balance(rows, "flagged_fraud")
    assert balance["positive"] == 2
    assert balance["negative"] == 2
    assert balance["positive_rate"] == 0.5
