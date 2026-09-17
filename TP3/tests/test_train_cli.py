"""Integration tests for the `tp3 train` CLI subcommand."""

import json
import subprocess
from pathlib import Path


def test_train_cli_xor_success(tp3_bin: Path, tmp_path: Path):
    config_path = Path("configs/xor_example.json")
    assert config_path.is_file()

    out_dir = tmp_path / "runs"
    res = subprocess.run(
        [str(tp3_bin), "train", "--config", str(config_path), "--out", str(out_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.returncode == 0
    assert "epoch=200" in res.stderr

    json_files = list(out_dir.glob("*.json"))
    assert len(json_files) == 1

    data = json.loads(json_files[0].read_text())
    assert len(data["loss_per_epoch"]) == 200
    assert data["loss_per_epoch"][-1] < data["loss_per_epoch"][0]
    assert data["loss"] == "mse"
    assert data["optimizer"] == "sgd"
    assert data["wall_time_seconds"] > 0.0


def test_train_cli_invalid_config_content(tp3_bin: Path, tmp_path: Path):
    bad_config = tmp_path / "bad.json"
    bad_config.write_text(
        json.dumps(
            {
                "format_version": 1,
                "model_type": "bogus",
                "activation": "tanh",
                "loss": "mse",
                "optimizer": "sgd",
                "learning_rate": 0.1,
                "epochs": 10,
                "seed": 42,
                "dataset_kind": "validation",
                "dataset_case": "xor",
            }
        )
    )

    res = subprocess.run(
        [str(tp3_bin), "train", "--config", str(bad_config)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode in (1, 2)
    assert "bogus" in res.stderr


def test_train_cli_missing_config_file(tp3_bin: Path, tmp_path: Path):
    missing_config = tmp_path / "nonexistent.json"
    res = subprocess.run(
        [str(tp3_bin), "train", "--config", str(missing_config)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode in (1, 2)
    assert res.returncode not in (134, 139)
    assert "cannot open" in res.stderr or "error" in res.stderr


def test_train_cli_save_and_resume(tp3_bin: Path, tmp_path: Path):
    model_path = tmp_path / "saved_model.json"
    cfg1_path = tmp_path / "cfg1.json"
    cfg1_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "model_type": "mlp",
                "layer_sizes": [2, 2, 1],
                "activation": "tanh",
                "loss": "mse",
                "optimizer": "sgd",
                "learning_rate": 0.1,
                "epochs": 30,
                "seed": 42,
                "dataset_kind": "validation",
                "dataset_case": "xor",
            }
        )
    )

    out_dir = tmp_path / "runs_resume"
    res1 = subprocess.run(
        [str(tp3_bin), "train", "--config", str(cfg1_path), "--out", str(out_dir), "--save-model", str(model_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res1.returncode == 0
    assert model_path.is_file()

    cfg2_path = tmp_path / "cfg2.json"
    cfg2_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "model_type": "mlp",
                "layer_sizes": [2, 2, 1],
                "activation": "tanh",
                "loss": "mse",
                "optimizer": "sgd",
                "learning_rate": 0.1,
                "epochs": 70,
                "seed": 42,
                "dataset_kind": "validation",
                "dataset_case": "xor",
            }
        )
    )

    res2 = subprocess.run(
        [str(tp3_bin), "train", "--config", str(cfg2_path), "--out", str(out_dir), "--resume-from", str(model_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res2.returncode == 0
    json_files = sorted(out_dir.glob("*.json"))
    assert len(json_files) == 2
    resumed_data = json.loads(json_files[-1].read_text())
    assert len(resumed_data["loss_per_epoch"]) == 70


def test_train_cli_csv_plain(tp3_bin: Path, tmp_path: Path):
    csv_file = tmp_path / "plain_data.csv"
    csv_file.write_text("x1,x2,y\n0.1,0.2,0.5\n0.3,0.4,0.8\n-0.1,0.5,0.2\n")

    cfg_path = tmp_path / "plain_cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "model_type": "mlp",
                "layer_sizes": [2, 2, 1],
                "activation": "tanh",
                "loss": "mse",
                "optimizer": "sgd",
                "learning_rate": 0.05,
                "epochs": 10,
                "seed": 42,
                "dataset_kind": "csv",
                "dataset_format": "plain",
                "dataset_path": str(csv_file),
                "dataset_target_column": "y",
            }
        )
    )

    out_dir = tmp_path / "runs_plain"
    res = subprocess.run(
        [str(tp3_bin), "train", "--config", str(cfg_path), "--out", str(out_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.returncode == 0
    json_files = list(out_dir.glob("*.json"))
    assert len(json_files) == 1
    data = json.loads(json_files[0].read_text())
    assert data["dataset_path"] == str(csv_file)
    assert len(data["loss_per_epoch"]) == 10


def test_train_cli_csv_digits(tp3_bin: Path, tmp_path: Path):
    digits_file = tmp_path / "digits_small.csv"
    digits_file.write_text("label,image\n0,\"[0.1, 0.2, 0.3]\"\n1,\"[0.4, 0.5, 0.6]\"\n")

    cfg_path = tmp_path / "digits_cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "model_type": "mlp",
                "layer_sizes": [3, 2],
                "activation": "tanh",
                "use_softmax_output": True,
                "loss": "cross_entropy",
                "optimizer": "sgd",
                "learning_rate": 0.05,
                "epochs": 5,
                "seed": 42,
                "dataset_kind": "csv",
                "dataset_format": "digits",
                "dataset_path": str(digits_file),
            }
        )
    )

    out_dir = tmp_path / "runs_digits"
    res = subprocess.run(
        [str(tp3_bin), "train", "--config", str(cfg_path), "--out", str(out_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.returncode == 0
    json_files = list(out_dir.glob("*.json"))
    assert len(json_files) == 1
    data = json.loads(json_files[0].read_text())
    assert data["dataset_path"] == str(digits_file)
    assert data["loss"] == "cross_entropy"
    assert len(data["loss_per_epoch"]) == 5
