import csv
import json
import math
import random
from pathlib import Path

RAW_CSV = Path("data/data and documentation/fraud_dataset.csv")
FEATURE_COLUMNS = [
    "timestamp",
    "amount_usd",
    "quantity_purchased",
    "session_duration_seconds",
    "days_since_last_purchase",
    "account_age_days",
    "device_screen_resolution",
    "time_since_last_login_s",
    "items_viewed_before_purchase"
]
TARGET_COLUMN = "big_model_fraud_probability"
LABEL_COLUMN = "flagged_fraud"
OUT_DIR = Path("data/derived")
SEED = 42
TEST_FRACTION = 0.2

def split_indices(n: int, seed: int, test_fraction: float) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_test = round(n * test_fraction)
    test_idx = sorted(indices[:n_test])
    train_idx = sorted(indices[n_test:])
    return train_idx, test_idx

def standardize_fit(rows: list[dict], columns: list[str]) -> dict:
    stats = {}
    n = len(rows)
    for col in columns:
        vals = [float(r[col]) for r in rows]
        mean = sum(vals) / n if n > 0 else 0.0
        var = sum((x - mean) ** 2 for x in vals) / n if n > 0 else 0.0
        std = math.sqrt(var)
        if std == 0.0 or math.isclose(std, 0.0, abs_tol=1e-12):
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats

def standardize_apply(row: dict, stats: dict, columns: list[str]) -> dict:
    scaled = {}
    for col in columns:
        val = float(row[col])
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        scaled[col] = (val - mean) / std
    return scaled

def write_feature_csv(path: Path, rows: list[dict], stats: dict, feature_cols=None, target_col=None):
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    if target_col is None:
        target_col = TARGET_COLUMN
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(feature_cols) + [target_col]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            scaled = standardize_apply(r, stats, feature_cols)
            row_vals = [str(scaled[col]) for col in feature_cols]
            row_vals.append(str(r[target_col]))
            writer.writerow(row_vals)

def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw dataset missing: {RAW_CSV}")

    with open(RAW_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Assign row_id = 0..7499 by enumeration order (never re-sort)
    for i, r in enumerate(all_rows):
        r["row_id"] = i

    train_idx, test_idx = split_indices(len(all_rows), SEED, TEST_FRACTION)
    train_set = set(train_idx)
    test_set = set(test_idx)

    train_rows = [all_rows[i] for i in train_idx]
    test_rows = [all_rows[i] for i in test_idx]

    # Compute standardize_fit ONLY on train rows' FEATURE_COLUMNS
    stats = standardize_fit(train_rows, FEATURE_COLUMNS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. fraud_train.csv
    write_feature_csv(OUT_DIR / "fraud_train.csv", train_rows, stats)
    # 2. fraud_test.csv
    write_feature_csv(OUT_DIR / "fraud_test.csv", test_rows, stats)
    # 3. fraud_full.csv (ALL rows in original order 0..7499)
    write_feature_csv(OUT_DIR / "fraud_full.csv", all_rows, stats)

    # 4. fraud_scaler.json
    scaler_info = {
        "feature_columns": FEATURE_COLUMNS,
        "stats": stats,
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "n_train": len(train_idx),
        "n_test": len(test_idx)
    }
    with open(OUT_DIR / "fraud_scaler.json", "w", encoding="utf-8") as f:
        json.dump(scaler_info, f, indent=2)

    # 5. fraud_labels.csv (original order 0..7499)
    labels_csv_path = OUT_DIR / "fraud_labels.csv"
    with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", LABEL_COLUMN, "split"])
        for r in all_rows:
            rid = r["row_id"]
            split = "train" if rid in train_set else "test"
            writer.writerow([rid, r[LABEL_COLUMN], split])

    print(f"Preprocessing complete: train={len(train_rows)}, test={len(test_rows)}, total={len(all_rows)}")

if __name__ == "__main__":
    main()
