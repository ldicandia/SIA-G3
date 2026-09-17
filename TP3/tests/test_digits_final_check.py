import glob
import json
from pathlib import Path
import pytest


def test_digits_test_csv_exactly_one_script_reference():
    """Standing guard for DIGIT-02: digits_test.csv must be opened by exactly one script."""
    script_files = list(Path("scripts").rglob("*.py"))
    matches = []
    for sf in script_files:
        content = sf.read_text(encoding="utf-8")
        if "digits_test.csv" in content:
            matches.append(sf.name)

    assert len(matches) == 1, f"digits_test.csv matched in unexpected scripts: {matches}"
    assert matches[0] == "digits_compare.py"


def test_final_test_metrics_if_present():
    path = Path("docs/ejercicio2/final_test_metrics.json")
    if not path.exists():
        pytest.skip("final_test_metrics.json not yet generated")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("n_predictions") == 2497
    assert data.get("accuracy", 0.0) > 0.10

    recalls = data.get("per_class_recall", {})
    assert len(recalls) == 10
    for c in range(10):
        val = recalls.get(str(c))
        assert val is not None, f"Recall for class {c} was unexpectedly None in test metrics"
