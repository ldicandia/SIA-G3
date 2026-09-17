import glob
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from scripts.ejercicio2.digits_metrics import accuracy, per_class_recall, load_class_predictions
    from scripts.ejercicio2.digits_train_variant import train_and_harvest
except ModuleNotFoundError:
    from digits_metrics import accuracy, per_class_recall, load_class_predictions
    from digits_train_variant import train_and_harvest

NEW_LR_VARIANTS = [0.01, 2.0]
BASE_LAYER_SIZES = [784, 32, 10]
BASE_ACTIVATION = "sigmoid"
BASE_OPTIMIZER = "sgd"
BASE_EPOCHS = 10


def summarize_variant(
    run_name: str, train_run_json: Path | str, val_run_json: Path | str
) -> dict:
    train_path = Path(train_run_json)
    val_path = Path(val_run_json)

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    hp = train_data.get("hyperparameters", {})
    lr = hp.get("learning_rate")
    epochs = hp.get("epochs")
    architecture = hp.get("architecture", [])
    optimizer = train_data.get("optimizer") or hp.get("optimizer", "sgd")

    loss_per_epoch = train_data.get("loss_per_epoch", [])
    has_non_finite = any(v is None for v in loss_per_epoch)

    y_true, y_pred = load_class_predictions(val_path)
    val_acc = accuracy(y_true, y_pred)
    recalls = per_class_recall(y_true, y_pred, num_classes=10)

    # Stringify recall keys for clean JSON
    str_recalls = {str(k): v for k, v in recalls.items()}

    return {
        "run_name": run_name,
        "learning_rate": lr,
        "optimizer": optimizer,
        "epochs": epochs,
        "layer_sizes": architecture,
        "val_accuracy": val_acc,
        "per_class_recall": str_recalls,
        "has_non_finite_loss": has_non_finite,
        "train_run_json": str(train_path),
        "val_run_json": str(val_path),
    }


def main():
    variants = []

    # 1. New learning rate variants
    for lr in NEW_LR_VARIANTS:
        run_name = f"lr_{lr}"
        print(f"Training variant: {run_name} (lr={lr})...")
        train_json, val_json = train_and_harvest(
            run_name=run_name,
            layer_sizes=BASE_LAYER_SIZES,
            activation=BASE_ACTIVATION,
            optimizer=BASE_OPTIMIZER,
            learning_rate=lr,
            train_csv="data/derived/digits/train.csv",
            val_csv="data/derived/digits/val.csv",
            base_out_dir="runs/digits",
            epochs=BASE_EPOCHS,
        )
        variants.append(summarize_variant(run_name, train_json, val_json))

    # 2. Baseline variant (lr=0.05) - reuse existing runs without retraining
    baseline_train_runs = [
        p for p in Path("runs/digits/baseline").glob("*.json") if p.name != "model.json"
    ]
    baseline_val_runs = [
        p for p in Path("runs/digits/baseline_val").glob("*.json") if p.name != "model.json"
    ]
    if not baseline_train_runs or not baseline_val_runs:
        raise FileNotFoundError("Baseline runs missing in runs/digits/baseline*")

    baseline_train = sorted(baseline_train_runs)[-1]
    baseline_val = sorted(baseline_val_runs)[-1]
    variants.append(summarize_variant("baseline", baseline_train, baseline_val))

    # Sort variants by learning_rate ascending
    variants.sort(key=lambda v: v["learning_rate"])

    out_path = Path("docs/ejercicio2/lr_variants.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"variants": variants}, f, indent=2)

    print(f"Learning rate sweep complete! Saved to {out_path}")
    for v in variants:
        print(
            f"  lr={v['learning_rate']}: val_acc={v['val_accuracy']:.4f}, non_finite={v['has_non_finite_loss']}"
        )


if __name__ == "__main__":
    main()
