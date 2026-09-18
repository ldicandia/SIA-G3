import json
from pathlib import Path

import pytest


def test_heldout_csv_exactly_one_script_reference():
    """Standing guard for ACC-01: heldout.csv must be opened by exactly one script.

    Matches the FULL literal dataset path (`data/derived/more_digits/heldout.csv`),
    not the bare filename: `scripts/ejercicio3/more_digits_preprocess.py` (Plan
    07-01, already committed) legitimately constructs `OUT_DIR / "heldout.csv"`
    to WRITE the split and immediately read it back once for a row-count/
    label-integrity self-check during split creation - it never trains or
    evaluates a model against heldout.csv's contents, so it never assembles the
    full path as one literal string. Only `more_digits_compare.py`'s
    `--final-check` path treats heldout.csv as a dataset input to `tp3 train`.
    """
    script_files = list(Path("scripts").rglob("*.py"))
    matches = []
    for sf in script_files:
        content = sf.read_text(encoding="utf-8")
        if "data/derived/more_digits/heldout.csv" in content:
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
