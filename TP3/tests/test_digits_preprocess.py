from pathlib import Path
from scripts.digits_preprocess import (
    stratified_split_indices,
    label_counts,
    read_raw_lines,
    NUM_CLASSES,
    RAW_TRAIN_CSV,
)


def test_stratified_split_indices_properties():
    # Synthetic labels with missing class 8 and skewed distribution
    labels = [0] * 50 + [1] * 30 + [5] * 10 + [9] * 20
    n = len(labels)
    train_idx, val_idx = stratified_split_indices(labels, seed=42, val_fraction=0.2)

    assert set(train_idx).isdisjoint(set(val_idx))
    assert sorted(train_idx + val_idx) == list(range(n))

    # Repeatability
    t2, v2 = stratified_split_indices(labels, seed=42, val_fraction=0.2)
    assert train_idx == t2
    assert val_idx == v2

    # Different seed yields different order / selections if possible
    # Check class ratios
    val_labels = [labels[i] for i in val_idx]
    # Class 0: 50 * 0.2 = 10
    assert val_labels.count(0) == 10
    # Class 1: 30 * 0.2 = 6
    assert val_labels.count(1) == 6
    # Class 5: 10 * 0.2 = 2
    assert val_labels.count(5) == 2
    # Class 9: 20 * 0.2 = 4
    assert val_labels.count(9) == 4


def test_label_counts():
    lines = ["0,[0.1]", "1,[0.2]", "1,[0.3]", "5,[0.4]"]
    counts = label_counts(lines)
    assert len(counts) == NUM_CLASSES
    assert counts["0"] == 1
    assert counts["1"] == 2
    assert counts["5"] == 1
    assert counts["8"] == 0
    for d in range(NUM_CLASSES):
        assert str(d) in counts


def test_train_val_csv_partition_if_generated():
    train_csv = Path("data/derived/digits/train.csv")
    val_csv = Path("data/derived/digits/val.csv")
    if train_csv.exists() and val_csv.exists():
        _, train_lines = read_raw_lines(train_csv)
        _, val_lines = read_raw_lines(val_csv)
        assert len(train_lines) + len(val_lines) == 12449
