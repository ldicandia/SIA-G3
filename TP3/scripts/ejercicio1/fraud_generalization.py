import csv
import json
from pathlib import Path
import random
import statistics
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from scripts.ejercicio1.fraud_preprocess import (
        RAW_CSV,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        LABEL_COLUMN,
        standardize_fit,
        write_feature_csv
    )
    from scripts.ejercicio1.fraud_train_variant import build_config, run_tp3_train
    from scripts.ejercicio1.fraud_metrics import precision_recall_f1
except ModuleNotFoundError:
    from fraud_preprocess import (
        RAW_CSV,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        LABEL_COLUMN,
        standardize_fit,
        write_feature_csv
    )
    from fraud_train_variant import build_config, run_tp3_train
    from fraud_metrics import precision_recall_f1

K_FOLDS = 5
BASE_SEED = 42

def kfold_partition(n: int, k: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    chunk_size = n // k
    return [indices[i * chunk_size : (i + 1) * chunk_size] for i in range(k)]

def main():
    model_comp_path = Path("docs/ejercicio1/model_comparison.json")
    if not model_comp_path.exists():
        raise FileNotFoundError(f"Model comparison missing: {model_comp_path}")

    model_comp = json.loads(model_comp_path.read_text(encoding="utf-8"))
    selected_model = model_comp["selected_model"]
    print(f"Running 5-fold cross validation for selected model: {selected_model}")

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw dataset missing: {RAW_CSV}")

    with open(RAW_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    for i, r in enumerate(all_rows):
        r["row_id"] = i

    n = len(all_rows)
    chunks = kfold_partition(n, K_FOLDS, BASE_SEED)

    per_fold_metrics = []
    pooled_oof = []

    for fold in range(K_FOLDS):
        print(f"\n--- Fold {fold} ---")
        test_idx = chunks[fold]
        train_idx = [idx for j in range(K_FOLDS) if j != fold for idx in chunks[j]]

        train_rows = [all_rows[i] for i in train_idx]
        test_rows = [all_rows[i] for i in test_idx]

        # Fit standardization strictly on this fold's train rows
        fold_stats = standardize_fit(train_rows, FEATURE_COLUMNS)

        fold_dir = Path(f"data/derived/folds/fold_{fold}")
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_csv = fold_dir / "train.csv"
        test_csv = fold_dir / "test.csv"
        labels_csv = fold_dir / "labels.csv"

        write_feature_csv(train_csv, train_rows, fold_stats)
        write_feature_csv(test_csv, test_rows, fold_stats)

        with open(labels_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row_id", LABEL_COLUMN])
            for r in test_rows:
                writer.writerow([r["row_id"], r[LABEL_COLUMN]])

        # Train on fold train.csv
        train_out_dir = Path(f"runs/fraud/fold_{fold}")
        train_out_dir.mkdir(parents=True, exist_ok=True)
        train_cfg = build_config(
            activation=selected_model,
            dataset_path=train_csv,
            output_dir=train_out_dir,
            epochs=50,
            learning_rate=0.01,
            seed=BASE_SEED
        )
        train_cfg_path = fold_dir / "train_config.json"
        with open(train_cfg_path, "w", encoding="utf-8") as f:
            json.dump(train_cfg, f, indent=2)

        model_path = train_out_dir / "model.json"
        run_tp3_train("build/tp3", train_cfg_path, save_model=model_path)

        # Harvest on fold test.csv (zero additional epochs)
        test_out_dir = Path(f"runs/fraud/fold_{fold}_test")
        test_out_dir.mkdir(parents=True, exist_ok=True)
        test_cfg = dict(train_cfg)
        test_cfg["dataset_path"] = str(test_csv)
        test_cfg["output_dir"] = str(test_out_dir)
        test_cfg_path = fold_dir / "test_config.json"
        with open(test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(test_cfg, f, indent=2)

        run_tp3_train("build/tp3", test_cfg_path, resume_from=model_path)

        test_runs = [p for p in test_out_dir.glob("*.json") if p.name != "model.json"]
        if not test_runs:
            raise FileNotFoundError(f"No test run json found in {test_out_dir}")
        latest_test_run = sorted(test_runs)[-1]

        run_data = json.loads(latest_test_run.read_text(encoding="utf-8"))
        predictions = run_data["predictions"]

        if len(predictions) != len(test_rows):
            raise ValueError(f"Fold {fold}: Expected {len(test_rows)} predictions, got {len(predictions)}")

        y_true = [int(r[LABEL_COLUMN]) for r in test_rows]
        y_prob = [max(0.0, min(1.0, float(p["predicted"]))) for p in predictions]

        for yt, yp in zip(y_true, y_prob):
            pooled_oof.append([yt, yp])

        fold_eval = precision_recall_f1(y_true, y_prob, threshold=0.5)
        per_fold_metrics.append({
            "fold": fold,
            "precision": fold_eval["precision"],
            "recall": fold_eval["recall"],
            "f1": fold_eval["f1"],
            "n_test": len(test_rows),
            "n_positive": sum(y_true)
        })
        print(f"Fold {fold} (threshold=0.5): Precision={fold_eval['precision']:.4f}, Recall={fold_eval['recall']:.4f}, F1={fold_eval['f1']:.4f}")

    if len(pooled_oof) != n:
        raise ValueError(f"Pooled OOF length mismatch: expected {n}, got {len(pooled_oof)}")

    precisions = [m["precision"] for m in per_fold_metrics]
    recalls = [m["recall"] for m in per_fold_metrics]
    f1s = [m["f1"] for m in per_fold_metrics]

    mean_metrics = {
        "precision": statistics.mean(precisions),
        "recall": statistics.mean(recalls),
        "f1": statistics.mean(f1s)
    }
    std_metrics = {
        "precision": statistics.pstdev(precisions),
        "recall": statistics.pstdev(recalls),
        "f1": statistics.pstdev(f1s)
    }

    baseline_acc = 1.0 - (869.0 / 7500.0)

    gen_results = {
        "selected_model": selected_model,
        "k_folds": K_FOLDS,
        "per_fold": per_fold_metrics,
        "mean": mean_metrics,
        "std": std_metrics,
        "baseline_accuracy_always_negative": baseline_acc
    }

    gen_out_path = Path("docs/ejercicio1/generalization_metrics.json")
    gen_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gen_out_path, "w", encoding="utf-8") as f:
        json.dump(gen_results, f, indent=2)

    oof_out_path = Path("data/derived/oof_predictions.json")
    oof_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(oof_out_path, "w", encoding="utf-8") as f:
        json.dump(pooled_oof, f)

    print(f"\nGeneralization metrics saved to {gen_out_path}")
    print(f"OOF predictions ({len(pooled_oof)} pairs) saved to {oof_out_path}")
    print(f"Mean Precision: {mean_metrics['precision']:.4f} +/- {std_metrics['precision']:.4f}")
    print(f"Mean Recall:    {mean_metrics['recall']:.4f} +/- {std_metrics['recall']:.4f}")
    print(f"Mean F1:        {mean_metrics['f1']:.4f} +/- {std_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
