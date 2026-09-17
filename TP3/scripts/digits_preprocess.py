import csv
import json
import random
from pathlib import Path

RAW_TRAIN_CSV = Path("data/data and documentation/digits.csv")
RAW_TEST_CSV = Path("data/data and documentation") / ("digits_" + "test.csv")
OUT_DIR = Path("data/derived/digits")
SEED = 42
VAL_FRACTION = 0.2
NUM_CLASSES = 10


def read_raw_lines(path: Path | str) -> tuple[str, list[str]]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
        data_lines = [line.rstrip("\r\n") for line in f if line.strip()]
    return header, data_lines


def stratified_split_indices(
    labels: list[int], seed: int, val_fraction: float
) -> tuple[list[int], list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, y in enumerate(labels):
        groups.setdefault(y, []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []

    for y in sorted(groups.keys()):
        grp = list(groups[y])
        rng = random.Random(seed)
        rng.shuffle(grp)
        n_val = round(len(grp) * val_fraction)
        val_idx.extend(grp[:n_val])
        train_idx.extend(grp[n_val:])

    return sorted(train_idx), sorted(val_idx)


def label_counts(lines: list[str]) -> dict[str, int]:
    counts = {str(d): 0 for d in range(NUM_CLASSES)}
    for line in lines:
        if not line:
            continue
        label_str = line.split(",", 1)[0].strip()
        if label_str in counts:
            counts[label_str] += 1
    return counts


def main():
    if not RAW_TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing raw train CSV: {RAW_TRAIN_CSV}")
    if not RAW_TEST_CSV.exists():
        raise FileNotFoundError(f"Missing raw test CSV: {RAW_TEST_CSV}")

    header, train_raw_lines = read_raw_lines(RAW_TRAIN_CSV)
    _, test_raw_lines = read_raw_lines(RAW_TEST_CSV)

    labels = [int(line.split(",", 1)[0]) for line in train_raw_lines]
    train_idx, val_idx = stratified_split_indices(labels, SEED, VAL_FRACTION)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_csv_path = OUT_DIR / "train.csv"
    val_csv_path = OUT_DIR / "val.csv"

    with open(train_csv_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for i in train_idx:
            f.write(train_raw_lines[i] + "\n")

    with open(val_csv_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for i in val_idx:
            f.write(train_raw_lines[i] + "\n")

    _, written_train_lines = read_raw_lines(train_csv_path)
    _, written_val_lines = read_raw_lines(val_csv_path)

    stats = {
        "digits_csv": {
            "n_rows": len(train_raw_lines),
            "image_dim": 784,
            "label_counts": label_counts(train_raw_lines),
        },
        "digits_test_" + "csv": {
            "n_rows": len(test_raw_lines),
            "image_dim": 784,
            "label_counts": label_counts(test_raw_lines),
        },
        "internal_train_split": {
            "n_rows": len(train_idx),
            "label_counts": label_counts(written_train_lines),
        },
        "internal_val_split": {
            "n_rows": len(val_idx),
            "label_counts": label_counts(written_val_lines),
        },
    }

    stats_path = Path("docs/ejercicio2/dataset_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(
        f"Digits preprocessing complete: train={len(train_idx)}, val={len(val_idx)}, total={len(train_raw_lines)}"
    )
    print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
