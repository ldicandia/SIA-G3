import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from scripts.ejercicio1.fraud_metrics import sweep, best_threshold_by_f1
except ModuleNotFoundError:
    from fraud_metrics import sweep, best_threshold_by_f1

def main():
    oof_path = Path("data/derived/oof_predictions.json")
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions file not found: {oof_path}")

    oof_data = json.loads(oof_path.read_text(encoding="utf-8"))
    y_true = [int(p[0]) for p in oof_data]
    y_prob = [float(p[1]) for p in oof_data]

    thresholds = [round(0.05 * i, 2) for i in range(1, 20)]
    grid_results = sweep(y_true, y_prob, thresholds)
    best = best_threshold_by_f1(grid_results)

    default_05 = next(r for r in grid_results if abs(r["threshold"] - 0.5) < 1e-6)

    sweep_artifact = {
        "grid": grid_results,
        "recommended_threshold": best["threshold"],
        "recommended_metrics": {
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"]
        },
        "default_0.5_metrics": default_05,
        "selection_rule": "argmax F1 over a 0.05-step grid from 0.05 to 0.95; no business cost weighting was provided, so F1 is the neutral default (not a default-0.5 threshold)"
    }

    out_path = Path("docs/ejercicio1/threshold_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sweep_artifact, f, indent=2)

    print(f"Threshold sweep written to {out_path}")
    print(f"Recommended threshold: {best['threshold']} (F1={best['f1']:.4f}, Precision={best['precision']:.4f}, Recall={best['recall']:.4f})")
    print(f"Default 0.5 threshold:  0.50 (F1={default_05['f1']:.4f}, Precision={default_05['precision']:.4f}, Recall={default_05['recall']:.4f})")

if __name__ == "__main__":
    main()
