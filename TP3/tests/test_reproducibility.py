"""End-to-end: a run is reproducible from its seed (ENG-05), asserted on the JSON bytes (VAL-09),
and re-running into the same --out directory overwrites atomically (ENG-07)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

CASES = ("and", "linear", "tanh", "xor")


@pytest.mark.parametrize("case", CASES)
def test_same_seed_same_json(run_validate, case: str) -> None:
    a = run_validate(case, seed=7)
    b = run_validate(case, seed=7)

    assert a.path != b.path
    assert a.path.read_bytes() == b.path.read_bytes()
    assert a.data["final_weights"] == b.data["final_weights"]
    assert a.data["bias"] == b.data["bias"]
    assert a.data["loss_per_epoch"] == b.data["loss_per_epoch"]

    if case == "xor":
        for stem in ("xor_221", "xor_2321", "xor_step"):
            fa = (a.path.parent / f"{stem}.json").read_bytes()
            fb = (b.path.parent / f"{stem}.json").read_bytes()
            assert fa == fb


def test_different_seed_changes_first_epoch_loss(run_validate) -> None:
    seed7 = run_validate("linear", seed=7).data
    seed8 = run_validate("linear", seed=8).data
    assert seed7["loss_per_epoch"][0] != seed8["loss_per_epoch"][0]


def test_rerun_into_same_out_dir_overwrites_atomically(tp3_bin: Path, run_validate, tmp_path: Path) -> None:
    same = tmp_path / "same"
    for _ in range(2):
        subprocess.run(
            [str(tp3_bin), "validate", "and", "--seed", "42", "--out", str(same)],
            check=True,
            capture_output=True,
            text=True,
        )

    assert sorted(p.name for p in same.iterdir()) == ["and.json"]
    assert not list(same.glob("*.tmp"))
    fresh = run_validate("and", seed=42)
    assert (same / "and.json").read_bytes() == fresh.path.read_bytes()
