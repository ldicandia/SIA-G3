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

NEW_OPTIMIZER_VARIANTS = ["momentum", "adam"]
BASE_LAYER_SIZES = [784, 32, 10]
BASE_ACTIVATION = "sigmoid"
BASE_LR = 0.05
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
    str_recalls = {str(k): v for k, v in recalls.items()}

    return {
        "run_name": run_name,
        "optimizer": optimizer,
        "learning_rate": lr,
        "epochs": epochs,
        "architecture": architecture,
        "layer_sizes": architecture,
        "val_accuracy": val_acc,
        "per_class_recall": str_recalls,
        "has_non_finite_loss": has_non_finite,
        "train_run_json": str(train_path),
        "val_run_json": str(val_path),
    }


def main():
    variants_dict = {}

    # 1. Baseline variant ("sgd") - reuse existing without retraining
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
    variants_dict["sgd"] = summarize_variant(
        "baseline", baseline_train, baseline_val
    )

    # 2. New optimizer variants ("momentum", "adam")
    for opt in NEW_OPTIMIZER_VARIANTS:
        run_name = f"opt_{opt}"
        print(f"Training optimizer variant: {run_name} (optimizer={opt})...")
        train_json, val_json = train_and_harvest(
            run_name=run_name,
            layer_sizes=BASE_LAYER_SIZES,
            activation=BASE_ACTIVATION,
            optimizer=opt,
            learning_rate=BASE_LR,
            train_csv="data/derived/digits/train.csv",
            val_csv="data/derived/digits/val.csv",
            base_out_dir="runs/digits",
            epochs=BASE_EPOCHS,
        )
        variants_dict[opt] = summarize_variant(run_name, train_json, val_json)

    # Order fixed: sgd, momentum, adam
    ordered_variants = [variants_dict["sgd"], variants_dict["momentum"], variants_dict["adam"]]

    out_path = Path("docs/ejercicio2/optimizer_variants.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"variants": ordered_variants}, f, indent=2)

    print(f"Optimizer sweep complete! Saved to {out_path}")
    for v in ordered_variants:
        print(
            f"  optimizer={v['optimizer']}: val_acc={v['val_accuracy']:.4f}, loss_clean={not v['has_non_finite_loss']}"
        )


if __name__ == "__main__":
    main()
