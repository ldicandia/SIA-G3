import json
from pathlib import Path


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}"
        )
    if not y_true:
        return 0.0
    matches = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return matches / len(y_true)


def per_class_recall(
    y_true: list[int], y_pred: list[int], num_classes: int = 10
) -> dict[int, float | None]:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}"
        )

    recalls: dict[int, float | None] = {}
    for c in range(num_classes):
        total_class = sum(1 for yt in y_true if yt == c)
        if total_class == 0:
            recalls[c] = None
        else:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
            recalls[c] = tp / total_class

    return recalls


def load_class_predictions(run_json_path: Path | str) -> tuple[list[int], list[int]]:
    path = Path(run_json_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    y_true: list[int] = []
    y_pred: list[int] = []

    for i, p in enumerate(predictions):
        if "predicted_class" not in p or "expected_class" not in p:
            raise ValueError(
                f"Prediction {i} in {path} lacks predicted_class or expected_class"
            )
        pc = p["predicted_class"]
        ec = p["expected_class"]
        if pc < 0 or ec < 0:
            raise ValueError(
                f"Prediction {i} in {path} has invalid class sentinel: predicted={pc}, expected={ec}"
            )
        y_pred.append(pc)
        y_true.append(ec)

    return y_true, y_pred


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = sorted(Path("runs/digits/baseline_val").glob("*.json"))[-1]

    y_true, y_pred = load_class_predictions(json_path)
    acc = accuracy(y_true, y_pred)
    recalls = per_class_recall(y_true, y_pred)
    print(f"File: {json_path}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("Per-class recall:")
    for c in range(10):
        val = recalls[c]
        formatted = f"{val:.4f} ({val*100:.2f}%)" if val is not None else "None (0 samples)"
        print(f"  Class {c}: {formatted}")
