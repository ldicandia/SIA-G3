import json
from pathlib import Path

import pytest


def test_heldout_csv_exactly_one_script_reference():
    """Standing guard for ACC-01: heldout.csv must be opened by exactly one script."""
    script_files = list(Path("scripts").rglob("*.py"))
    matches = []
    for sf in script_files:
        content = sf.read_text(encoding="utf-8")
        if "heldout.csv" in content:
            matches.append(sf.name)

    assert len(matches) == 1, f"heldout.csv matched in unexpected scripts: {matches}"
    assert matches[0] == "more_digits_compare.py"


def test_final_heldout_metrics_if_present():
    path = Path("docs/ejercicio3/final_heldout_metrics.json")
    if not path.exists():
        pytest.skip("final_heldout_metrics.json not yet generated")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    heldout_csv = Path("data/derived/more_digits/heldout.csv")
    lines = [
        line
        for line in heldout_csv.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][1:]
    assert data.get("n_predictions") == len(lines)
    assert isinstance(data.get("accuracy"), float)

    recalls = data.get("per_class_recall", {})
    assert len(recalls) == 10
    for c in range(10):
        val = recalls.get(str(c))
        assert val is not None, f"Recall for class {c} was unexpectedly None in final heldout metrics"

    assert data.get("final_choice_type") in ("single_model", "ensemble")
    assert data.get("meets_98_percent_target") == (data.get("accuracy") >= 0.98)
