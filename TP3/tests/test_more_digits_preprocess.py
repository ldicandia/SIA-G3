from pathlib import Path

from scripts.ejercicio2.digits_preprocess import read_raw_lines
from scripts.ejercicio3.more_digits_preprocess import stratified_three_way_split_indices


def test_stratified_three_way_split_indices_properties():
    # Synthetic labels with a skewed distribution, mirroring digits_preprocess's own test shape.
    labels = [0] * 50 + [1] * 30 + [5] * 10 + [9] * 20
    n = len(labels)
    train_idx, val_idx, heldout_idx = stratified_three_way_split_indices(
        labels, seed=42, val_fraction=0.2, heldout_fraction=0.2
    )

    # Pairwise disjoint.
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx).isdisjoint(set(heldout_idx))
    assert set(val_idx).isdisjoint(set(heldout_idx))

    # Full partition.
    assert sorted(train_idx + val_idx + heldout_idx) == list(range(n))

    # Repeatability.
    t2, v2, h2 = stratified_three_way_split_indices(
        labels, seed=42, val_fraction=0.2, heldout_fraction=0.2
    )
    assert train_idx == t2
    assert val_idx == v2
    assert heldout_idx == h2

    # Per-class heldout/val counts (round(count * fraction) each).
    heldout_labels = [labels[i] for i in heldout_idx]
    val_labels = [labels[i] for i in val_idx]

    # Class 0: 50 rows -> round(50*0.2)=10 heldout, 10 val, 30 train
    assert heldout_labels.count(0) == 10
    assert val_labels.count(0) == 10
    # Class 1: 30 rows -> round(30*0.2)=6 heldout, 6 val, 18 train
    assert heldout_labels.count(1) == 6
    assert val_labels.count(1) == 6
    # Class 5: 10 rows -> round(10*0.2)=2 heldout, 2 val, 6 train
    assert heldout_labels.count(5) == 2
    assert val_labels.count(5) == 2
    # Class 9: 20 rows -> round(20*0.2)=4 heldout, 4 val, 12 train
    assert heldout_labels.count(9) == 4
    assert val_labels.count(9) == 4


def test_stratified_three_way_split_indices_no_class_dropped():
    # A class with only 1 row must still land somewhere in the partition, never raise.
    labels = [0] * 5 + [3] * 1
    train_idx, val_idx, heldout_idx = stratified_three_way_split_indices(
        labels, seed=7, val_fraction=0.2, heldout_fraction=0.2
    )
    assert sorted(train_idx + val_idx + heldout_idx) == list(range(len(labels)))


def test_train_val_heldout_csv_partition_if_generated():
    train_csv = Path("data/derived/more_digits/train.csv")
    val_csv = Path("data/derived/more_digits/val.csv")
    heldout_csv = Path("data/derived/more_digits/heldout.csv")
    if train_csv.exists() and val_csv.exists() and heldout_csv.exists():
        _, train_lines = read_raw_lines(train_csv)
        _, val_lines = read_raw_lines(val_csv)
        _, heldout_lines = read_raw_lines(heldout_csv)
        assert len(train_lines) + len(val_lines) + len(heldout_lines) == 15741

        train_set = set(train_lines)
        val_set = set(val_lines)
        heldout_set = set(heldout_lines)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(heldout_set)
        assert val_set.isdisjoint(heldout_set)
