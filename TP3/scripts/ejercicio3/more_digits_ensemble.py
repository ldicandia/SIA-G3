import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ejercicio2.digits_metrics import accuracy, per_class_recall, load_class_predictions

# Plan 07-02's four already-trained seed-variant models (baseline seed=42 +
# seed_7/123/2026). No new tp3 train invocation occurs in this plan - this
# script only combines those four models' already-harvested val predictions.
SEED_VARIANTS_PATH = Path("docs/ejercicio3/seed_variants.json")
OUT_PATH = Path("docs/ejercicio3/ensemble_variants.json")


def majority_vote(predictions_per_member: list[list[int]]) -> list[int]:
    """Combine each member's per-row predicted class via hard majority vote.

    For each row index i, collects `[member[i] for member in predictions_per_member]`
    and returns the most common value via `collections.Counter`. Ties are broken
    deterministically to the smallest class value among the tied classes.
    """
    if not predictions_per_member:
        return []

    n_rows = len(predictions_per_member[0])
    combined: list[int] = []
    for i in range(n_rows):
        votes = [member[i] for member in predictions_per_member]
        counts = Counter(votes)
        max_count = max(counts.values())
        winner = min(cls for cls, count in counts.items() if count == max_count)
        combined.append(winner)
    return combined


def assert_rows_aligned(y_true_by_member: dict[str, list[int]]) -> None:
    """Fail loudly if any member's val expected_class sequence diverges.

    Ensembling via row-index-based majority voting is only valid if every
    member was harvested against the exact same val.csv rows in the exact
    same order. Raises AssertionError naming the first differing member.
    """
    items = list(y_true_by_member.items())
    if not items:
        return

    first_name, first_y_true = items[0]
    for name, y_true in items[1:]:
        if y_true != first_y_true:
            raise AssertionError(
                f"Row-alignment guard failed: member '{name}'s val expected_class sequence "
                f"does not match member '{first_name}'s - ensembling via row-index majority "
                f"vote requires every member to be harvested against the exact same val.csv "
                f"rows in the exact same order."
            )


def main():
    with open(SEED_VARIANTS_PATH, "r", encoding="utf-8") as f:
        seed_data = json.load(f)
    members = seed_data["variants"]
    run_names = [v["run_name"] for v in members]

    y_true_by_member: dict[str, list[int]] = {}
    y_pred_by_member: dict[str, list[int]] = {}
    member_val_accuracies: dict[str, float] = {}

    for variant in members:
        run_name = variant["run_name"]
        y_true, y_pred = load_class_predictions(variant["val_run_json"])
        y_true_by_member[run_name] = y_true
        y_pred_by_member[run_name] = y_pred
        member_val_accuracies[run_name] = accuracy(y_true, y_pred)

    assert_rows_aligned(y_true_by_member)

    shared_y_true = y_true_by_member[run_names[0]]
    predictions_per_member = [y_pred_by_member[name] for name in run_names]

    ensemble_pred = majority_vote(predictions_per_member)
    ensemble_val_accuracy = accuracy(shared_y_true, ensemble_pred)
    ensemble_recalls = per_class_recall(shared_y_true, ensemble_pred, num_classes=10)
    ensemble_per_class_recall = {str(k): v for k, v in ensemble_recalls.items()}

    best_single_member_val_accuracy = max(member_val_accuracies.values())
    ensemble_beats_best_single_member = ensemble_val_accuracy > best_single_member_val_accuracy

    result = {
        "members": run_names,
        "member_val_accuracies": member_val_accuracies,
        "ensemble_val_accuracy": ensemble_val_accuracy,
        "ensemble_per_class_recall": ensemble_per_class_recall,
        "best_single_member_val_accuracy": best_single_member_val_accuracy,
        "ensemble_beats_best_single_member": ensemble_beats_best_single_member,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Ensemble complete! Saved to {OUT_PATH}")
    print(f"  members: {run_names}")
    print(f"  ensemble_val_accuracy={ensemble_val_accuracy:.4f}")
    print(f"  best_single_member_val_accuracy={best_single_member_val_accuracy:.4f}")
    print(f"  ensemble_beats_best_single_member={ensemble_beats_best_single_member}")


if __name__ == "__main__":
    main()
