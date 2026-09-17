import csv
import json
from pathlib import Path
import pytest

def test_sigmoid_predictions_bounded():
    sig_files = sorted(Path("runs/fraud/sigmoid_full").glob("*.json"))
    assert len(sig_files) > 0, "No sigmoid harvest run json found"
    sig_data = json.loads(sig_files[-1].read_text(encoding="utf-8"))
    predictions = sig_data["predictions"]
    assert len(predictions) == 7500

    for i, p in enumerate(predictions):
        val = p["predicted"]
        assert 0.0 <= val <= 1.0, f"Sigmoid prediction at row {i} out of bounds: {val}"

def test_prediction_alignment_with_raw_csv():
    raw_csv_path = Path("data/data and documentation/fraud_dataset.csv")
    assert raw_csv_path.exists()
    with open(raw_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_expected = [float(r["big_model_fraud_probability"]) for r in reader]

    assert len(raw_expected) == 7500

    id_files = sorted(Path("runs/fraud/identity_full").glob("*.json"))
    assert len(id_files) > 0, "No identity harvest run json found"
    id_data = json.loads(id_files[-1].read_text(encoding="utf-8"))
    predictions = id_data["predictions"]
    assert len(predictions) == 7500

    for i in range(7500):
        pred_exp = predictions[i]["expected"]
        raw_exp = raw_expected[i]
        assert abs(pred_exp - raw_exp) < 1e-6, f"Mismatch at index {i}: pred_exp={pred_exp}, raw_exp={raw_exp}"

def test_linear_out_of_bounds_count():
    id_files = sorted(Path("runs/fraud/identity_full").glob("*.json"))
    assert len(id_files) > 0
    id_data = json.loads(id_files[-1].read_text(encoding="utf-8"))
    preds = [p["predicted"] for p in id_data["predictions"]]
    below_0 = [p for p in preds if p < 0.0]
    above_1 = [p for p in preds if p > 1.0]
    out_of_bounds = below_0 + above_1
    print(f"\nLinear out of bounds count: {len(out_of_bounds)} / {len(preds)} (below 0: {len(below_0)}, above 1: {len(above_1)}, min: {min(preds):.6f}, max: {max(preds):.6f})")
