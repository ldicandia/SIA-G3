import argparse
import glob
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from scripts.ejercicio2.digits_metrics import accuracy, per_class_recall, load_class_predictions
    from scripts.ejercicio2.digits_train_variant import build_config, run_tp3_train, train_and_harvest
except ModuleNotFoundError:
    from digits_metrics import accuracy, per_class_recall, load_class_predictions
    from digits_train_variant import build_config, run_tp3_train, train_and_harvest


def select_best(variants: list[dict], key: str = "val_accuracy") -> dict:
    if not variants:
        raise ValueError("Variants list cannot be empty")

    def sort_key(v):
        val = v.get(key, 0.0)
        params = v.get("n_params")
        if params is None:
            params = float("inf")
        lr = v.get("learning_rate")
        if lr is None:
            lr = float("inf")
        opt = v.get("optimizer") or "zzzz"
        return (-val, params, lr, opt)

    return min(variants, key=sort_key)


def main():
    lr_file = Path("docs/ejercicio2/lr_variants.json")
    arch_file = Path("docs/ejercicio2/architecture_variants.json")
    opt_file = Path("docs/ejercicio2/optimizer_variants.json")

    with open(lr_file, "r", encoding="utf-8") as f:
        lr_data = json.load(f)
    with open(arch_file, "r", encoding="utf-8") as f:
        arch_data = json.load(f)
    with open(opt_file, "r", encoding="utf-8") as f:
        opt_data = json.load(f)

    best_lr = select_best(lr_data["variants"])
    best_arch = select_best(arch_data["variants"])
    best_opt = select_best(opt_data["variants"])

    combined = {
        "layer_sizes": best_arch["layer_sizes"],
        "learning_rate": best_lr["learning_rate"],
        "optimizer": best_opt["optimizer"],
        "activation": "sigmoid",
        "epochs": 10,
    }

    print("Composed combined-best configuration:")
    print(f"  layer_sizes: {combined['layer_sizes']}")
    print(f"  learning_rate: {combined['learning_rate']}")
    print(f"  optimizer: {combined['optimizer']}")

    print("Training combined_best model...")
    train_json, val_json = train_and_harvest(
        run_name="combined_best",
        layer_sizes=combined["layer_sizes"],
        activation=combined["activation"],
        optimizer=combined["optimizer"],
        learning_rate=combined["learning_rate"],
        train_csv="data/derived/digits/train.csv",
        val_csv="data/derived/digits/val.csv",
        base_out_dir="runs/digits",
        epochs=combined["epochs"],
    )

    y_true, y_pred = load_class_predictions(val_json)
    val_acc = accuracy(y_true, y_pred)
    recalls = per_class_recall(y_true, y_pred, num_classes=10)
    str_recalls = {str(k): v for k, v in recalls.items()}

    out_data = {
        "lr_variants": lr_data["variants"],
        "architecture_variants": arch_data["variants"],
        "optimizer_variants": opt_data["variants"],
        "selected": {
            **combined,
            "val_accuracy": val_acc,
            "per_class_recall": str_recalls,
            "train_run_json": str(train_json),
            "val_run_json": str(val_json),
        },
        "selection_rule": (
            "maximize val_accuracy; ties broken by fewer n_params, "
            "then lower learning_rate, then alphabetically-first optimizer name"
        ),
    }

    out_path = Path("docs/ejercicio2/variant_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"Comparison saved to {out_path}")
    print(f"Combined-best validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")


def final_check(tp3_bin: Path | str = "build/tp3"):
    comp_file = Path("docs/ejercicio2/variant_comparison.json")
    if not comp_file.exists():
        raise FileNotFoundError(
            "variant_comparison.json missing. Run main comparison first."
        )

    with open(comp_file, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    selected = comp_data["selected"]
    model_path = Path("runs/digits/combined_best/model.json")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model missing at {model_path}")

    # DIGIT-02 single production check: digits_test.csv
    test_csv = Path("data/data and documentation/digits_test.csv")
    out_dir = Path("runs/digits/combined_best_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dir = Path("data/derived/digits/configs")
    config_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(
        layer_sizes=selected["layer_sizes"],
        activation=selected.get("activation", "sigmoid"),
        optimizer=selected["optimizer"],
        learning_rate=selected["learning_rate"],
        dataset_path=test_csv,
        output_dir=out_dir,
        epochs=selected.get("epochs", 10),
        seed=42,
    )
    cfg_path = config_dir / "combined_best_test.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Executing single final production check on {test_csv}...")
    run_tp3_train(tp3_bin, cfg_path, resume_from=model_path)

    test_runs = [p for p in out_dir.glob("*.json") if p.name != "model.json"]
    if not test_runs:
        raise FileNotFoundError(f"No run JSON generated in {out_dir}")
    test_run_json = sorted(test_runs)[-1]

    y_true, y_pred = load_class_predictions(test_run_json)
    acc = accuracy(y_true, y_pred)
    recalls = per_class_recall(y_true, y_pred, num_classes=10)
    str_recalls = {str(k): v for k, v in recalls.items()}

    final_metrics = {
        "n_predictions": len(y_true),
        "accuracy": acc,
        "per_class_recall": str_recalls,
        "config": {
            "layer_sizes": selected["layer_sizes"],
            "learning_rate": selected["learning_rate"],
            "optimizer": selected["optimizer"],
        },
        "test_run_json": str(test_run_json),
    }

    final_metrics_path = Path("docs/ejercicio2/final_test_metrics.json")
    with open(final_metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Final test metrics written to {final_metrics_path}")
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("Per-class recall:")
    for c in range(10):
        val = str_recalls[str(c)]
        print(f"  Class {c}: {val:.4f} ({val*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare digits MLP variants and execute final test check"
    )
    parser.add_argument(
        "--final-check",
        action="store_true",
        help="Execute single final test check on digits_test.csv",
    )
    args = parser.parse_args()

    if args.final_check:
        final_check()
    else:
        main()
