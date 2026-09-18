import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ejercicio2.digits_metrics import accuracy, per_class_recall, load_class_predictions
from scripts.ejercicio2.digits_train_variant import train_and_harvest


def load_ejercicio2_best_config(
    path: Path | str = Path("docs/ejercicio2/variant_comparison.json"),
) -> dict:
    """Load Ejercicio 2's own combined-best config verbatim (never retyped).

    Raises ValueError naming the missing key if `selected` is absent, or if
    `selected` is missing any of layer_sizes/learning_rate/optimizer/
    activation/epochs.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "selected" not in data:
        raise ValueError(f"{path} is missing required key 'selected'")

    selected = data["selected"]
    required_keys = ("layer_sizes", "learning_rate", "optimizer", "activation", "epochs")
    for key in required_keys:
        if key not in selected:
            raise ValueError(f"{path}'s 'selected' config is missing required key '{key}'")

    return selected


def run_more_data_baseline(tp3_bin: Path | str = "build/tp3") -> tuple[Path, Path]:
    """Re-run Ejercicio 2's exact combined-best config, unmodified, on more_digits.csv."""
    cfg = load_ejercicio2_best_config()
    train_json, val_json = train_and_harvest(
        run_name="more_data_baseline",
        layer_sizes=cfg["layer_sizes"],
        activation=cfg["activation"],
        optimizer=cfg["optimizer"],
        learning_rate=cfg["learning_rate"],
        train_csv="data/derived/more_digits/train.csv",
        val_csv="data/derived/more_digits/val.csv",
        base_out_dir="runs/more_digits",
        epochs=cfg["epochs"],
        tp3_bin=tp3_bin,
    )
    return train_json, val_json


def main():
    cfg = load_ejercicio2_best_config()
    train_json, val_json = run_more_data_baseline()

    y_true, y_pred = load_class_predictions(val_json)
    val_acc = accuracy(y_true, y_pred)
    recalls = per_class_recall(y_true, y_pred, num_classes=10)
    str_recalls = {str(k): v for k, v in recalls.items()}

    ejercicio2_val_accuracy = cfg["val_accuracy"]
    delta = val_acc - ejercicio2_val_accuracy

    out_data = {
        "ejercicio2_best_config": cfg,
        "ejercicio2_val_accuracy": ejercicio2_val_accuracy,
        "more_data_baseline": {
            "val_accuracy": val_acc,
            "per_class_recall": str_recalls,
            "train_run_json": str(train_json),
            "val_run_json": str(val_json),
        },
        "delta_val_accuracy": delta,
    }

    out_path = Path("docs/ejercicio3/more_data_baseline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"More-data baseline val accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")
    print(f"Ejercicio 2 val accuracy: {ejercicio2_val_accuracy:.4f}")
    print(f"Delta: {delta:+.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
