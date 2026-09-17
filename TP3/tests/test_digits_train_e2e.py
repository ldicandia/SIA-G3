import glob
import pytest
from pathlib import Path
from scripts.ejercicio2.digits_metrics import accuracy, per_class_recall, load_class_predictions
from scripts.ejercicio2.digits_train_variant import build_config


def test_metrics_accuracy():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 1]
    assert pytest.approx(accuracy(y_true, y_pred)) == 2 / 3


def test_metrics_per_class_recall():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 1]
    recalls = per_class_recall(y_true, y_pred, num_classes=4)
    assert recalls[0] == 1.0
    assert recalls[1] == 1.0
    assert recalls[2] == 0.0
    assert recalls[3] is None  # Class 3 has 0 instances in y_true


def test_build_config():
    cfg = build_config(
        layer_sizes=[784, 32, 10],
        activation="sigmoid",
        optimizer="sgd",
        learning_rate=0.05,
        dataset_path="data/derived/digits/train.csv",
        output_dir="runs/digits/baseline",
        epochs=10,
        seed=42,
    )
    assert cfg["model_type"] == "mlp"
    assert cfg["use_softmax_output"] is True
    assert cfg["loss"] == "cross_entropy"
    assert cfg["optimizer"] == "sgd"
    assert cfg["learning_rate"] == 0.05
    assert cfg["epochs"] == 10
    assert cfg["dataset_format"] == "digits"
    assert "dataset_target_column" not in cfg


def test_baseline_val_run_if_present():
    runs = sorted(glob.glob("runs/digits/baseline_val/*.json"))
    if not runs:
        pytest.skip("Baseline validation run not yet executed")
    latest = runs[-1]
    y_true, y_pred = load_class_predictions(latest)
    acc = accuracy(y_true, y_pred)
    recalls = per_class_recall(y_true, y_pred, num_classes=10)

    assert acc > 0.10
    assert recalls[8] is None
