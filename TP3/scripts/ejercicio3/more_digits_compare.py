import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ejercicio2.digits_compare import select_best
from scripts.ejercicio2.digits_metrics import accuracy, per_class_recall, load_class_predictions
from scripts.ejercicio2.digits_train_variant import build_config, run_tp3_train
from scripts.ejercicio3.more_digits_ensemble import majority_vote

ARCH_VARIANTS_PATH = Path("docs/ejercicio3/architecture_variants.json")
SEED_VARIANTS_PATH = Path("docs/ejercicio3/seed_variants.json")
ENSEMBLE_VARIANTS_PATH = Path("docs/ejercicio3/ensemble_variants.json")
COMPARISON_PATH = Path("docs/ejercicio3/variant_comparison.json")
FINAL_METRICS_PATH = Path("docs/ejercicio3/final_heldout_metrics.json")
HELDOUT_CSV = Path("data/derived/more_digits/heldout.csv")

SELECTION_RULE = (
    "maximize val_accuracy across the merged architecture+seed variant pool "
    "(Phase 6's select_best tie-break: fewer n_params, then lower learning_rate, "
    "then alphabetically-first optimizer); the ensemble replaces the single-model "
    "choice only if its val_accuracy is strictly higher, ties preferring the "
    "single model for lower deployment complexity"
)


def choose_final(best_individual: dict, ensemble: dict) -> dict:
    """Decide between the best single model and the seed-variant ensemble.

    The ensemble wins only if its val_accuracy is strictly higher than the
    best individual's. Ties (and any lower ensemble result) prefer the single
    model, for lower deployment complexity and a deterministic outcome.
    """
    if ensemble["ensemble_val_accuracy"] > best_individual["val_accuracy"]:
        return {
            "type": "ensemble",
            "members": ensemble["members"],
            "val_accuracy": ensemble["ensemble_val_accuracy"],
        }
    return {
        "type": "single_model",
        "run_name": best_individual["run_name"],
        "layer_sizes": best_individual["layer_sizes"],
        "learning_rate": best_individual["learning_rate"],
        "optimizer": best_individual["optimizer"],
        "activation": best_individual.get("activation", "sigmoid"),
        "epochs": best_individual.get("epochs", 10),
        "val_accuracy": best_individual["val_accuracy"],
    }


def _merge_individual_variants(architecture_variants: list[dict], seed_variants: list[dict]) -> list[dict]:
    """Merge the architecture and seed variant pools, de-duplicated by run_name.

    The shared "more_data_baseline" entry appears in both source files - keep
    only the first occurrence. Seed variants record their architecture under
    the key "architecture" rather than "layer_sizes"; normalize so every
    merged entry carries "layer_sizes" regardless of source, since
    choose_final's single_model branch reads that key unconditionally.
    """
    merged: list[dict] = []
    seen_run_names: set[str] = set()
    for variant in architecture_variants + seed_variants:
        run_name = variant["run_name"]
        if run_name in seen_run_names:
            continue
        seen_run_names.add(run_name)
        normalized = dict(variant)
        if "layer_sizes" not in normalized and "architecture" in normalized:
            normalized["layer_sizes"] = normalized["architecture"]
        merged.append(normalized)
    return merged


def main():
    with open(ARCH_VARIANTS_PATH, "r", encoding="utf-8") as f:
        arch_data = json.load(f)
    with open(SEED_VARIANTS_PATH, "r", encoding="utf-8") as f:
        seed_data = json.load(f)
    with open(ENSEMBLE_VARIANTS_PATH, "r", encoding="utf-8") as f:
        ensemble_data = json.load(f)

    all_individual = _merge_individual_variants(arch_data["variants"], seed_data["variants"])
    best_individual = select_best(all_individual)
    final_choice = choose_final(best_individual, ensemble_data)

    out_data = {
        "architecture_variants": arch_data["variants"],
        "seed_variants": seed_data["variants"],
        "ensemble_variant": ensemble_data,
        "best_individual": best_individual,
        "final_choice": final_choice,
        "selection_rule": SELECTION_RULE,
    }

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"Comparison saved to {COMPARISON_PATH}")
    print(f"Best individual: {best_individual['run_name']} (val_accuracy={best_individual['val_accuracy']:.4f})")
    print(f"Final choice: type={final_choice['type']}, val_accuracy={final_choice['val_accuracy']:.4f}")


def final_check(tp3_bin: Path | str = "build/tp3"):
    if not COMPARISON_PATH.exists():
        raise FileNotFoundError(
            "variant_comparison.json missing. Run the main comparison first."
        )

    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    final_choice = comp_data["final_choice"]

    config_dir = Path("data/derived/more_digits/configs")
    config_dir.mkdir(parents=True, exist_ok=True)

    if final_choice["type"] == "single_model":
        run_name = final_choice["run_name"]
        model_path = Path(f"runs/more_digits/{run_name}/model.json")
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model missing at {model_path}")

        out_dir = Path("runs/more_digits/final_heldout_single")
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = build_config(
            layer_sizes=final_choice["layer_sizes"],
            activation=final_choice.get("activation", "sigmoid"),
            optimizer=final_choice["optimizer"],
            learning_rate=final_choice["learning_rate"],
            dataset_path=HELDOUT_CSV,
            output_dir=out_dir,
            epochs=final_choice.get("epochs", 10),
            seed=42,
        )
        cfg_path = config_dir / "final_heldout_single.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        print(f"Executing single final production check on {HELDOUT_CSV}...")
        run_tp3_train(tp3_bin, cfg_path, resume_from=model_path)

        run_files = [p for p in out_dir.glob("*.json") if p.name != "model.json"]
        if not run_files:
            raise FileNotFoundError(f"No run JSON generated in {out_dir}")
        run_json = sorted(run_files)[-1]

        y_true, y_pred = load_class_predictions(run_json)
    else:
        members = final_choice["members"]
        y_true_by_member: dict[str, list[int]] = {}
        y_pred_by_member: dict[str, list[int]] = {}

        for run_name in members:
            model_path = Path(f"runs/more_digits/{run_name}/model.json")
            if not model_path.exists():
                raise FileNotFoundError(f"Trained model missing at {model_path}")

            out_dir = Path(f"runs/more_digits/final_heldout_{run_name}")
            out_dir.mkdir(parents=True, exist_ok=True)

            member_variant = None
            for v in comp_data["seed_variants"]:
                if v["run_name"] == run_name:
                    member_variant = v
                    break
            if member_variant is None:
                raise ValueError(f"Ensemble member '{run_name}' not found in seed_variants")

            cfg = build_config(
                layer_sizes=member_variant.get("layer_sizes", member_variant.get("architecture")),
                activation=member_variant.get("activation", "sigmoid"),
                optimizer=member_variant["optimizer"],
                learning_rate=member_variant["learning_rate"],
                dataset_path=HELDOUT_CSV,
                output_dir=out_dir,
                epochs=member_variant.get("epochs", 10),
                seed=42,
            )
            cfg_path = config_dir / f"final_heldout_{run_name}.json"
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            print(f"Executing final production check for ensemble member '{run_name}' on {HELDOUT_CSV}...")
            run_tp3_train(tp3_bin, cfg_path, resume_from=model_path)

            run_files = [p for p in out_dir.glob("*.json") if p.name != "model.json"]
            if not run_files:
                raise FileNotFoundError(f"No run JSON generated in {out_dir}")
            run_json = sorted(run_files)[-1]

            member_y_true, member_y_pred = load_class_predictions(run_json)
            y_true_by_member[run_name] = member_y_true
            y_pred_by_member[run_name] = member_y_pred

        y_true = y_true_by_member[members[0]]
        predictions_per_member = [y_pred_by_member[name] for name in members]
        y_pred = majority_vote(predictions_per_member)

    acc = accuracy(y_true, y_pred)
    recalls = per_class_recall(y_true, y_pred, num_classes=10)
    str_recalls = {str(k): v for k, v in recalls.items()}

    final_metrics = {
        "n_predictions": len(y_true),
        "accuracy": acc,
        "per_class_recall": str_recalls,
        "final_choice_type": final_choice["type"],
        "meets_98_percent_target": acc >= 0.98,
    }

    with open(FINAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Final heldout metrics written to {FINAL_METRICS_PATH}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Meets >=98% target: {final_metrics['meets_98_percent_target']}")
    print("Per-class recall:")
    for c in range(10):
        val = str_recalls[str(c)]
        formatted = f"{val:.4f} ({val*100:.2f}%)" if val is not None else "None (0 samples)"
        print(f"  Class {c}: {formatted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine ACC-03 technique results and execute the single final heldout check"
    )
    parser.add_argument(
        "--final-check",
        action="store_true",
        help="Execute the single final production check on data/derived/more_digits/heldout.csv",
    )
    args = parser.parse_args()

    if args.final_check:
        final_check()
    else:
        main()
