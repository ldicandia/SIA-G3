import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ejercicio2.digits_architecture_sweep import n_params
from scripts.ejercicio2.digits_metrics import accuracy, per_class_recall, load_class_predictions
from scripts.ejercicio2.digits_train_variant import train_and_harvest

# Baseline architecture [784, 32, 16, 10] is Plan 07-01's already-trained "more data"
# baseline, reused (never retrained) from runs/more_digits/more_data_baseline{,_val}.
NEW_ARCH_VARIANTS = [[784, 64, 10], [784, 64, 32, 10]]

BASELINE_JSON = Path("docs/ejercicio3/more_data_baseline.json")


def load_base_hyperparameters(path: Path | str = BASELINE_JSON) -> dict:
    """Load Plan 07-01's anchor config verbatim (never retyped)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["ejercicio2_best_config"]


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
    architecture = hp.get("architecture") or layer_sizes
    optimizer = train_data.get("optimizer")

    y_true, y_pred = load_class_predictions(val_path)
    val_acc = accuracy(y_true, y_pred)
    recalls = per_class_recall(y_true, y_pred, num_classes=10)
    str_recalls = {str(k): v for k, v in recalls.items()}

    return {
        "run_name": run_name,
        "layer_sizes": list(architecture),
        "n_params": n_params(layer_sizes),
        "learning_rate": lr,
        "optimizer": optimizer,
        "epochs": epochs,
        "val_accuracy": val_acc,
        "per_class_recall": str_recalls,
        "train_run_json": str(train_path),
        "val_run_json": str(val_path),
    }


def main():
    cfg = load_base_hyperparameters()
    base_lr = cfg["learning_rate"]
    base_optimizer = cfg["optimizer"]
    base_activation = cfg["activation"]
    base_epochs = cfg["epochs"]
    base_layer_sizes = cfg["layer_sizes"]

    variants = []

    for layer_sizes in NEW_ARCH_VARIANTS:
        hidden_str = "-".join(str(s) for s in layer_sizes[1:-1])
        run_name = f"arch_{hidden_str}"
        print(f"Training architecture variant: {run_name} {layer_sizes}...")
        train_json, val_json = train_and_harvest(
            run_name=run_name,
            layer_sizes=layer_sizes,
            activation=base_activation,
            optimizer=base_optimizer,
            learning_rate=base_lr,
            train_csv="data/derived/more_digits/train.csv",
            val_csv="data/derived/more_digits/val.csv",
            base_out_dir="runs/more_digits",
            epochs=base_epochs,
        )
        variants.append(summarize_variant(run_name, layer_sizes, train_json, val_json))

    # Baseline [784, 32, 16, 10] - reuse Plan 07-01's already-trained runs, never retrain.
    baseline_train_runs = [
        p for p in Path("runs/more_digits/more_data_baseline").glob("*.json")
        if p.name != "model.json"
    ]
    baseline_val_runs = [
        p for p in Path("runs/more_digits/more_data_baseline_val").glob("*.json")
        if p.name != "model.json"
    ]
    if not baseline_train_runs or not baseline_val_runs:
        raise FileNotFoundError(
            "Baseline runs missing in runs/more_digits/more_data_baseline* "
            "(run scripts/ejercicio3/more_digits_baseline.py first)"
        )
    baseline_train = sorted(baseline_train_runs)[-1]
    baseline_val = sorted(baseline_val_runs)[-1]
    variants.append(
        summarize_variant("more_data_baseline", base_layer_sizes, baseline_train, baseline_val)
    )

    variants.sort(key=lambda v: v["n_params"])

    out_path = Path("docs/ejercicio3/architecture_variants.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"variants": variants}, f, indent=2)

    print(f"Architecture sweep complete! Saved to {out_path}")
    for v in variants:
        print(f"  {v['layer_sizes']} (params={v['n_params']}): val_acc={v['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
