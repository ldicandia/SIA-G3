import argparse
import json
import subprocess
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_config(
    layer_sizes: list[int],
    activation: str,
    optimizer: str,
    learning_rate: float,
    dataset_path: Path | str,
    output_dir: Path | str,
    epochs: int = 10,
    seed: int = 42,
    progress_interval: int = 5,
) -> dict:
    return {
        "format_version": 1,
        "model_type": "mlp",
        "layer_sizes": layer_sizes,
        "activation": activation,
        "use_softmax_output": True,
        "loss": "cross_entropy",
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": seed,
        "dataset_kind": "csv",
        "dataset_format": "digits",
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "progress_interval_epochs": progress_interval,
    }


def run_tp3_train(
    tp3_bin: Path | str,
    config_path: Path | str,
    save_model: Path | str | None = None,
    resume_from: Path | str | None = None,
) -> None:
    cmd = [str(tp3_bin), "train", "--config", str(config_path)]
    if save_model is not None:
        cmd.extend(["--save-model", str(save_model)])
    if resume_from is not None:
        cmd.extend(["--resume-from", str(resume_from)])
    subprocess.run(cmd, check=True)


def train_and_harvest(
    run_name: str,
    layer_sizes: list[int],
    activation: str,
    optimizer: str,
    learning_rate: float,
    train_csv: Path | str,
    val_csv: Path | str,
    base_out_dir: Path | str = "runs/digits",
    epochs: int = 10,
    seed: int = 42,
    tp3_bin: Path | str = "build/tp3",
    config_dir: Path = Path("data/derived/digits/configs"),
) -> tuple[Path, Path]:
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    train_out_dir = Path(f"{base_out_dir}/{run_name}")
    val_out_dir = Path(f"{base_out_dir}/{run_name}_val")
    train_out_dir.mkdir(parents=True, exist_ok=True)
    val_out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = build_config(
        layer_sizes=layer_sizes,
        activation=activation,
        optimizer=optimizer,
        learning_rate=learning_rate,
        dataset_path=train_csv,
        output_dir=train_out_dir,
        epochs=epochs,
        seed=seed,
    )
    train_cfg_path = config_dir / f"{run_name}_train.json"
    with open(train_cfg_path, "w", encoding="utf-8") as f:
        json.dump(train_cfg, f, indent=2)

    model_save_path = train_out_dir / "model.json"
    run_tp3_train(tp3_bin, train_cfg_path, save_model=model_save_path)

    val_cfg = dict(train_cfg)
    val_cfg["dataset_path"] = str(val_csv)
    val_cfg["output_dir"] = str(val_out_dir)
    val_cfg_path = config_dir / f"{run_name}_val.json"
    with open(val_cfg_path, "w", encoding="utf-8") as f:
        json.dump(val_cfg, f, indent=2)

    run_tp3_train(tp3_bin, val_cfg_path, resume_from=model_save_path)

    train_runs = [p for p in train_out_dir.glob("*.json") if p.name != "model.json"]
    if not train_runs:
        raise FileNotFoundError(f"No run JSON found in {train_out_dir}")
    train_run_path = sorted(train_runs)[-1]

    val_runs = [p for p in val_out_dir.glob("*.json") if p.name != "model.json"]
    if not val_runs:
        raise FileNotFoundError(f"No run JSON found in {val_out_dir}")
    val_run_path = sorted(val_runs)[-1]

    return train_run_path, val_run_path


def main():
    parser = argparse.ArgumentParser(
        description="Train digits MLP variant and harvest validation predictions."
    )
    parser.add_argument("--run-name", required=True, help="Name of run")
    parser.add_argument(
        "--layer-sizes",
        required=True,
        help="Comma-separated layer sizes, e.g. 784,32,10",
    )
    parser.add_argument(
        "--activation",
        default="sigmoid",
        choices=["identity", "sigmoid", "tanh"],
        help="Hidden layer activation",
    )
    parser.add_argument(
        "--optimizer",
        default="sgd",
        choices=["sgd", "momentum", "adam"],
        help="Optimizer name",
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, help="Learning rate"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs to train")
    parser.add_argument(
        "--train-csv",
        default="data/derived/digits/train.csv",
        help="Training split CSV",
    )
    parser.add_argument(
        "--val-csv",
        default="data/derived/digits/val.csv",
        help="Validation split CSV",
    )
    parser.add_argument(
        "--base-out-dir", default="runs/digits", help="Base output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tp3-bin", default="build/tp3", help="tp3 executable")

    args = parser.parse_args()
    layers = [int(s.strip()) for s in args.layer_sizes.split(",") if s.strip()]

    train_run, val_run = train_and_harvest(
        run_name=args.run_name,
        layer_sizes=layers,
        activation=args.activation,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        base_out_dir=args.base_out_dir,
        epochs=args.epochs,
        seed=args.seed,
        tp3_bin=args.tp3_bin,
    )
    print(f"Train run JSON: {train_run}")
    print(f"Val run JSON: {val_run}")


if __name__ == "__main__":
    main()
