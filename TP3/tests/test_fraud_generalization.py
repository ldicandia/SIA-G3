import pytest
from scripts.ejercicio1.fraud_generalization import kfold_partition

def test_kfold_partition():
    n = 7500
    k = 5
    seed = 42
    chunks = kfold_partition(n, k, seed)
    assert len(chunks) == 5
    for chunk in chunks:
        assert len(chunk) == 1500

    all_indices = []
    for chunk in chunks:
        all_indices.extend(chunk)

    assert sorted(all_indices) == list(range(n))

    # Calling twice with same seed returns identical chunks
    chunks2 = kfold_partition(n, k, seed)
    assert chunks == chunks2

def test_fold_splits_sizes():
    n = 7500
    k = 5
    chunks = kfold_partition(n, k, 42)
    for i in range(k):
        test_idx = chunks[i]
        train_idx = [idx for j in range(k) if j != i for idx in chunks[j]]
        assert len(test_idx) == 1500
        assert len(train_idx) == 6000
        assert set(test_idx).isdisjoint(set(train_idx))
        assert sorted(test_idx + train_idx) == list(range(n))
