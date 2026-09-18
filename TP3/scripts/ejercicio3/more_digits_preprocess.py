import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ejercicio2.digits_preprocess import read_raw_lines, label_counts

RAW_CSV = Path("data/data and documentation/more_digits.csv")
OUT_DIR = Path("data/derived/more_digits")
SEED = 42
VAL_FRACTION = 0.2
HELDOUT_FRACTION = 0.2


def stratified_three_way_split_indices(
    labels: list[int], seed: int, val_fraction: float, heldout_fraction: float
) -> tuple[list[int], list[int], list[int]]:
    """Partition row indices, grouped by label, into disjoint train/val/heldout lists.

    Each label's group is shuffled independently with a fresh random.Random(seed)
    (mirroring digits_preprocess.stratified_split_indices's exact pattern), then
    split heldout-first, val-second, train-remainder. The union of the three lists
    is always a full, pairwise-disjoint partition of range(len(labels)).
    """
    groups: dict[int, list[int]] = {}
    for idx, y in enumerate(labels):
        groups.setdefault(y, []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    heldout_idx: list[int] = []

    for y in sorted(groups.keys()):
        grp = list(groups[y])
        rng = __import__("random").Random(seed)
        rng.shuffle(grp)
        n_heldout = round(len(grp) * heldout_fraction)
        n_val = round(len(grp) * val_fraction)
        heldout_idx.extend(grp[:n_heldout])
        val_idx.extend(grp[n_heldout : n_heldout + n_val])
        train_idx.extend(grp[n_heldout + n_val :])

    return sorted(train_idx), sorted(val_idx), sorted(heldout_idx)


def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing raw more_digits CSV: {RAW_CSV}")

    header, lines = read_raw_lines(RAW_CSV)
    labels = [int(line.split(",", 1)[0]) for line in lines]
    train_idx, val_idx, heldout_idx = stratified_three_way_split_indices(
        labels, SEED, VAL_FRACTION, HELDOUT_FRACTION
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_csv_path = OUT_DIR / "train.csv"
    val_csv_path = OUT_DIR / "val.csv"
    heldout_csv_path = OUT_DIR / "heldout.csv"

    def _write(path: Path, idx_list: list[int]) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(header + "\n")
            for i in idx_list:
                f.write(lines[i] + "\n")

    _write(train_csv_path, train_idx)
    _write(val_csv_path, val_idx)
    _write(heldout_csv_path, heldout_idx)

    _, written_train_lines = read_raw_lines(train_csv_path)
    _, written_val_lines = read_raw_lines(val_csv_path)
    _, written_heldout_lines = read_raw_lines(heldout_csv_path)

    digits_stats_path = Path("docs/ejercicio2/dataset_stats.json")
    with open(digits_stats_path, "r", encoding="utf-8") as f:
        digits_stats = json.load(f)
    digits_csv_stats = digits_stats["digits_csv"]

    stats = {
        "more_digits_csv": {
            "n_rows": len(lines),
            "image_dim": 784,
            "label_counts": label_counts(lines),
        },
        "digits_csv_for_comparison": digits_csv_stats,
        "internal_train_split": {
            "n_rows": len(train_idx),
            "label_counts": label_counts(written_train_lines),
        },
        "internal_val_split": {
            "n_rows": len(val_idx),
            "label_counts": label_counts(written_val_lines),
        },
        "heldout_split": {
            "n_rows": len(heldout_idx),
            "label_counts": label_counts(written_heldout_lines),
        },
        "row_count_delta_vs_digits_csv": len(lines) - digits_csv_stats["n_rows"],
    }

    stats_path = Path("docs/ejercicio3/dataset_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(
        f"more_digits preprocessing complete: train={len(train_idx)}, val={len(val_idx)}, "
        f"heldout={len(heldout_idx)}, total={len(lines)}"
    )
    print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
