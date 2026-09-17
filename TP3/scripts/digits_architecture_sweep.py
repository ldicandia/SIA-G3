import glob
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.digits_metrics import accuracy, per_class_recall, load_class_predictions
from scripts.digits_train_variant import train_and_harvest

NEW_ARCH_VARIANTS = [[784, 16, 10], [784, 32, 16, 10]]
BASE_LR = 0.05
BASE_ACTIVATION = "sigmoid"
BASE_OPTIMIZER = "sgd"
BASE_EPOCHS = 10


def n_params(layer_sizes: list[int]) -> int:
    return sum(
        layer_sizes[l] * layer_sizes[l + 1] + layer_sizes[l + 1]
        for l in range(len(layer_sizes) - 1)
    )


def summarize_variant(
    run_name: str,
    layer_sizes: list[int],
    train_run_json: Path | str,
    val_run_json: Path | str,
) -> dict:
    train_path = Path(train_run_json)
    val_path = Path(val_run_json)

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    hp = train_data.get("hyperparameters", {})
    lr = hp.get("learning_rate")
    epochs = hp.get("epochs")
    optimizer = train_data.get("optimizer") or hp.get("optimizer", "sgd")

    loss_per_epoch = train_data.get("loss_per_epoch", [])
    has_non_finite = any(v is None for v in loss_per_epoch)

    y_true, y_pred = load_class_predictions(val_path)
    val_acc = accuracy(y_true, y_pred)
    recalls = per_class_recall(y_true, y_pred, num_classes=10)
    str_recalls = {str(k): v for k, v in recalls.items()}

    return {
        "run_name": run_name,
        "layer_sizes": layer_sizes,
        "n_params": n_params(layer_sizes),
        "learning_rate": lr,
        "optimizer": optimizer,
        "epochs": epochs,
        "val_accuracy": val_acc,
        "per_class_recall": str_recalls,
        "has_non_finite_loss": has_non_finite,
        "train_run_json": str(train_path),
        "val_run_json": str(val_path),
    }


def main():
    variants = []

    # 1. New architecture variants
    for layer_sizes in NEW_ARCH_VARIANTS:
        hidden_str = "-".join(str(s) for s in layer_sizes[1:-1])
        run_name = f"arch_{hidden_str}"
        print(f"Training architecture variant: {run_name} {layer_sizes}...")
        train_json, val_json = train_and_harvest(
            run_name=run_name,
            layer_sizes=layer_sizes,
            activation=BASE_ACTIVATION,
            optimizer=BASE_OPTIMIZER,
            learning_rate=BASE_LR,
            train_csv="data/derived/digits/train.csv",
            val_csv="data/derived/digits/val.csv",
            base_out_dir="runs/digits",
            epochs=BASE_EPOCHS,
        )
        variants.append(
            summarize_variant(run_name, layer_sizes, train_json, val_json)
        )

    # 2. Baseline variant [784, 32, 10] - reuse without retraining
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
    variants.append(
        summarize_variant("baseline", [784, 32, 10], baseline_train, baseline_val)
    )

    # Sort variants by n_params ascending
    variants.sort(key=lambda v: v["n_params"])

    out_path = Path("docs/ejercicio2/architecture_variants.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"variants": variants}, f, indent=2)

    print(f"Architecture sweep complete! Saved to {out_path}")
    for v in variants:
        print(
            f"  {v['layer_sizes']} (params={v['n_params']}): val_acc={v['val_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
