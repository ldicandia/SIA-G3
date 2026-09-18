import glob

import pytest

from scripts.ejercicio2.digits_metrics import accuracy, load_class_predictions
from scripts.ejercicio3.more_digits_baseline import load_ejercicio2_best_config


def test_load_ejercicio2_best_config():
    cfg = load_ejercicio2_best_config()
    for key in ("layer_sizes", "learning_rate", "optimizer", "activation", "epochs"):
        assert key in cfg


def test_load_ejercicio2_best_config_missing_selected(tmp_path):
    bad_path = tmp_path / "no_selected.json"
    bad_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="selected"):
        load_ejercicio2_best_config(bad_path)


def test_load_ejercicio2_best_config_missing_required_key(tmp_path):
    bad_path = tmp_path / "missing_key.json"
    bad_path.write_text('{"selected": {"layer_sizes": [1, 2, 3]}}', encoding="utf-8")
    with pytest.raises(ValueError, match="learning_rate"):
        load_ejercicio2_best_config(bad_path)


def test_more_data_baseline_val_run_if_present():
    runs = sorted(glob.glob("runs/more_digits/more_data_baseline_val/*.json"))
    if not runs:
        pytest.skip("more_data_baseline validation run not yet executed")
    latest = runs[-1]
    y_true, y_pred = load_class_predictions(latest)
    acc = accuracy(y_true, y_pred)
    assert acc > 0.10
