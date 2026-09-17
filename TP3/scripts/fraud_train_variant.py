import argparse
import json
import subprocess
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from scripts.fraud_preprocess import TARGET_COLUMN
except ModuleNotFoundError:
    from fraud_preprocess import TARGET_COLUMN

def build_config(
    activation: str,
    dataset_path: Path | str,
    output_dir: Path | str,
    epochs: int = 50,
    learning_rate: float = 0.01,
    seed: int = 42
) -> dict:
    return {
        "format_version": 1,
        "model_type": "perceptron",
        "activation": activation,
        "loss": "mse",
        "optimizer": "sgd",
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": seed,
        "dataset_kind": "csv",
        "dataset_format": "plain",
        "dataset_path": str(dataset_path),
        "dataset_target_column": TARGET_COLUMN,
        "output_dir": str(output_dir),
        "progress_interval_epochs": 10
    }

def run_tp3_train(
    tp3_bin: Path | str,
    config_path: Path | str,
    save_model: Path | str | None = None,
    resume_from: Path | str | None = None
) -> None:
    cmd = [str(tp3_bin), "train", "--config", str(config_path)]
    if save_model is not None:
        cmd.extend(["--save-model", str(save_model)])
    if resume_from is not None:
        cmd.extend(["--resume-from", str(resume_from)])
    subprocess.run(cmd, check=True)

def train_and_harvest(
    activation: str,
    train_csv: Path | str,
    full_csv: Path | str,
    base_out_dir: Path | str = "runs/fraud",
    epochs: int = 50,
    learning_rate: float = 0.01,
    seed: int = 42,
    tp3_bin: Path | str = "build/tp3",
    config_dir: Path = Path("data/derived/configs")
) -> tuple[Path, Path]:
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    train_out_dir = Path(f"{base_out_dir}/{activation}")
    harvest_out_dir = Path(f"{base_out_dir}/{activation}_full")
    train_out_dir.mkdir(parents=True, exist_ok=True)
    harvest_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build and write train config
    train_cfg = build_config(activation, train_csv, train_out_dir, epochs, learning_rate, seed)
    train_cfg_path = config_dir / f"{activation}_train.json"
    with open(train_cfg_path, "w", encoding="utf-8") as f:
        json.dump(train_cfg, f, indent=2)

    model_save_path = train_out_dir / "model.json"
    run_tp3_train(tp3_bin, train_cfg_path, save_model=model_save_path)

    # 2. Build and write harvest config (identical except dataset_path and output_dir)
    harvest_cfg = dict(train_cfg)
    harvest_cfg["dataset_path"] = str(full_csv)
    harvest_cfg["output_dir"] = str(harvest_out_dir)
    harvest_cfg_path = config_dir / f"{activation}_full.json"
    with open(harvest_cfg_path, "w", encoding="utf-8") as f:
        json.dump(harvest_cfg, f, indent=2)

    run_tp3_train(tp3_bin, harvest_cfg_path, resume_from=model_save_path)

    # Find resulting run json files (ignoring model.json)
    train_runs = [p for p in train_out_dir.glob("*.json") if p.name != "model.json"]
    if not train_runs:
        raise FileNotFoundError(f"No run json found in {train_out_dir}")
    train_run_path = sorted(train_runs)[-1]

    harvest_runs = [p for p in harvest_out_dir.glob("*.json") if p.name != "model.json"]
    if not harvest_runs:
        raise FileNotFoundError(f"No run json found in {harvest_out_dir}")
    harvest_run_path = sorted(harvest_runs)[-1]

    return train_run_path, harvest_run_path

def main():
    parser = argparse.ArgumentParser(description="Train perceptron variant and harvest predictions on full dataset.")
    parser.add_argument("--activation", required=True, choices=["identity", "sigmoid"], help="Activation function")
    parser.add_argument("--train-csv", default="data/derived/fraud_train.csv", help="Path to training CSV")
    parser.add_argument("--full-csv", default="data/derived/fraud_full.csv", help="Path to full evaluation CSV")
    parser.add_argument("--base-out-dir", default="runs/fraud", help="Base directory for output runs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tp3-bin", default="build/tp3", help="Path to tp3 executable")

    args = parser.parse_args()
    train_run, harvest_run = train_and_harvest(
        activation=args.activation,
        train_csv=args.train_csv,
        full_csv=args.full_csv,
        base_out_dir=args.base_out_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        tp3_bin=args.tp3_bin
    )
    print(f"Train run JSON: {train_run}")
    print(f"Harvest run JSON: {harvest_run}")

if __name__ == "__main__":
    main()
