import csv
import json
from pathlib import Path

def split_predictions(predictions: list[dict], labels_rows: list[dict]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if len(predictions) != len(labels_rows):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(labels_rows)} labels rows")
    train_pairs = []
    test_pairs = []
    for p, lr in zip(predictions, labels_rows):
        pair = (float(p["expected"]), float(p["predicted"]))
        if lr["split"] == "train":
            train_pairs.append(pair)
        elif lr["split"] == "test":
            test_pairs.append(pair)
        else:
            raise ValueError(f"Unknown split: {lr['split']}")
    return train_pairs, test_pairs

def mse(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    return sum((exp - pred) ** 2 for exp, pred in pairs) / len(pairs)

def r2(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    mean_exp = sum(exp for exp, _ in pairs) / len(pairs)
    var_exp = sum((exp - mean_exp) ** 2 for exp, _ in pairs) / len(pairs)
    if var_exp == 0.0:
        return 0.0
    return 1.0 - (mse(pairs) / var_exp)

def clip_fraction(predictions: list[dict], low: float = 0.0, high: float = 1.0) -> float:
    if not predictions:
        return 0.0
    clipped = sum(1 for p in predictions if float(p["predicted"]) < low or float(p["predicted"]) > high)
    return clipped / len(predictions)

def saturation_fraction(predictions: list[dict], epsilon: float = 0.02) -> float:
    if not predictions:
        return 0.0
    saturated = sum(1 for p in predictions if float(p["predicted"]) <= epsilon or float(p["predicted"]) >= (1.0 - epsilon))
    return saturated / len(predictions)

def select_model(identity_metrics: dict, sigmoid_metrics: dict, r2_tie_threshold: float = 0.01) -> str:
    diff = identity_metrics["test_r2"] - sigmoid_metrics["test_r2"]
    if abs(diff) <= r2_tie_threshold:
        return "sigmoid"
    if diff > 0:
        return "identity"
    return "sigmoid"

def main():
    labels_csv_path = Path("data/derived/fraud_labels.csv")
    if not labels_csv_path.exists():
        raise FileNotFoundError(f"Labels CSV missing: {labels_csv_path}")

    with open(labels_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        labels_rows = list(reader)

    metrics = {}
    source_runs = {}

    for activation in ["identity", "sigmoid"]:
        run_dir = Path(f"runs/fraud/{activation}_full")
        run_files = sorted(run_dir.glob("*.json"))
        run_files = [p for p in run_files if p.name != "model.json"]
        if not run_files:
            raise FileNotFoundError(f"No run json found in {run_dir}")
        latest_run_file = run_files[-1]
        source_runs[activation] = str(latest_run_file)

        run_data = json.loads(latest_run_file.read_text(encoding="utf-8"))
        predictions = run_data["predictions"]

        train_pairs, test_pairs = split_predictions(predictions, labels_rows)
        train_mse = mse(train_pairs)
        test_mse = mse(test_pairs)
        test_r2 = r2(test_pairs)
        c_frac = clip_fraction(predictions, 0.0, 1.0)
        s_frac = saturation_fraction(predictions, 0.02)

        metrics[activation] = {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "test_r2": test_r2,
            "clip_fraction": c_frac,
            "saturation_fraction": s_frac
        }

    selected = select_model(metrics["identity"], metrics["sigmoid"])

    out_data = {
        "identity": metrics["identity"],
        "sigmoid": metrics["sigmoid"],
        "selected_model": selected,
        "selection_rule": "higher test_r2; tie (<=0.01) breaks to sigmoid to avoid post-hoc [0,1] clipping",
        "source_runs": source_runs
    }

    out_path = Path("docs/ejercicio1/model_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"Model comparison written to {out_path}")
    print(f"Selected model: {selected}")
    print(f"Identity metrics: {metrics['identity']}")
    print(f"Sigmoid metrics: {metrics['sigmoid']}")

if __name__ == "__main__":
    main()
