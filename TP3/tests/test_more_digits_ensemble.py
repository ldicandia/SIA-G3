import json
from pathlib import Path

import pytest

from scripts.ejercicio3.more_digits_ensemble import (
    assert_rows_aligned,
    majority_vote,
)


def test_majority_vote_basic():
    # row 0: all agree on 0; row 1: two vote 1, one votes 2 -> 1; row 2: two vote 2, one votes 1 -> 2
    result = majority_vote([[0, 1, 2], [0, 1, 1], [0, 2, 2]])
    assert result == [0, 1, 2]


def test_majority_vote_three_way_tie_breaks_to_smallest_class():
    # genuine 3-way tie for the single row -> smallest tied class (0)
    result = majority_vote([[0], [1], [2]])
    assert result == [0]


def test_majority_vote_unanimous():
    result = majority_vote([[5, 5], [5, 5], [5, 5]])
    assert result == [5, 5]


def test_assert_rows_aligned_passes_when_identical():
    # Should not raise
    assert_rows_aligned({"a": [0, 1, 2], "b": [0, 1, 2], "c": [0, 1, 2]})


def test_assert_rows_aligned_raises_naming_first_differing_member():
    with pytest.raises(AssertionError, match="b"):
        assert_rows_aligned({"a": [0, 1, 2], "b": [0, 1, 9], "c": [0, 1, 2]})


def test_ensemble_variants_json_if_present():
    """Standing guard: exactly the 4 seed-variant members, honest reporting."""
    path = Path("docs/ejercicio3/ensemble_variants.json")
    if not path.exists():
        pytest.skip("ensemble_variants.json not yet generated")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seed_path = Path("docs/ejercicio3/seed_variants.json")
    with open(seed_path, "r", encoding="utf-8") as f:
        seed_data = json.load(f)

    expected_members = sorted(v["run_name"] for v in seed_data["variants"])
    assert sorted(data["members"]) == expected_members, (
        f"expected exactly the 4 seed-variant members {expected_members}, got {sorted(data['members'])}"
    )

    assert isinstance(data["ensemble_val_accuracy"], float)
    assert data["ensemble_val_accuracy"] > 0.10

    assert isinstance(data["best_single_member_val_accuracy"], float)
    assert isinstance(data["ensemble_beats_best_single_member"], bool)
    assert data["ensemble_beats_best_single_member"] == (
        data["ensemble_val_accuracy"] > data["best_single_member_val_accuracy"]
    )
