import argparse
import csv
import json
from pathlib import Path

def compute_column_stats(rows: list[dict], column_name: str) -> dict:
    values = [float(r[column_name]) for r in rows if r.get(column_name) not in (None, "")]
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "count": len(values)
    }

def count_duplicate_rows(rows: list[dict]) -> int:
    seen = set()
    duplicates = 0
    for r in rows:
        t = tuple(sorted(r.items()))
        if t in seen:
            duplicates += 1
        else:
            seen.add(t)
    return duplicates

def class_balance(rows: list[dict], column_name: str) -> dict:
    pos = sum(1 for r in rows if int(float(r[column_name])) == 1)
    neg = sum(1 for r in rows if int(float(r[column_name])) == 0)
    total = pos + neg
    rate = pos / total if total > 0 else 0.0
    return {
        "positive": pos,
        "negative": neg,
        "positive_rate": rate
    }

def main():
    parser = argparse.ArgumentParser(description="Explore fraud dataset and compute descriptive statistics.")
    parser.add_argument("--in", dest="input_path", default="data/data and documentation/fraud_dataset.csv", help="Input CSV path")
    parser.add_argument("--stats-out", dest="stats_out", default="docs/ejercicio1/dataset_stats.json", help="Output JSON path")
    args = parser.parse_args()

    in_path = Path(args.input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    numeric_columns = [
        "timestamp", "amount_usd", "quantity_purchased", "session_duration_seconds",
        "days_since_last_purchase", "account_age_days", "device_screen_resolution",
        "time_since_last_login_s", "items_viewed_before_purchase",
        "big_model_fraud_probability", "flagged_fraud"
    ]

    columns_stats = {col: compute_column_stats(rows, col) for col in numeric_columns}
    dup_count = count_duplicate_rows(rows)
    balance = class_balance(rows, "flagged_fraud")

    stats = {
        "n_rows": len(rows),
        "n_cols": len(rows[0]) if rows else 0,
        "columns": columns_stats,
        "duplicates": dup_count,
        "flagged_fraud": balance
    }

    out_path = Path(args.stats_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats written to {out_path}")

if __name__ == "__main__":
    main()
