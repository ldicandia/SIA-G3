import json
from pathlib import Path
import pytest

def test_threshold_sweep_artifact():
    sweep_path = Path("docs/ejercicio1/threshold_sweep.json")
    assert sweep_path.exists()
    data = json.loads(sweep_path.read_text(encoding="utf-8"))
    assert len(data["grid"]) == 19
    assert "recommended_threshold" in data
    assert data["recommended_threshold"] == 0.9

def test_report_sections_present():
    report_path = Path("docs/ejercicio1/REPORT.md")
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "## 5. Generalization Study (FRAUD-05)" in content
    assert "## 6. Detection Threshold (FRAUD-06)" in content
    assert "## 7. Target Column & Knowledge-Distillation Note (FRAUD-07)" in content
    assert "big_model_fraud_probability" in content
    assert "no temperature-scaled soft-label matching or logit distillation is implemented" in content
