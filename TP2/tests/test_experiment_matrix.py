"""Pure matrix-cell generation tests (EXP-01/EXP-02): product_cells,
apply_overrides, and load_matrix_spec validation.

No process is spawned anywhere in this file -- that is `test_experiment_runner.py`'s job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tp2.experiments.matrix import (
    MatrixSpecError,
    apply_overrides,
    load_matrix_spec,
    product_cells,
)


# --- product_cells -----------------------------------------------------------


def test_product_cells_single_dimension_yields_two_labeled_cells() -> None:
    dims = {
        "selection": [
            ("elite", {"parents": {"method": "elite"}}),
            ("roulette", {"parents": {"method": "roulette"}}),
        ]
    }
    cells = list(product_cells(dims))
    assert len(cells) == 2
    ids = {cell.cell_id for cell in cells}
    assert ids == {"selection-elite", "selection-roulette"}


def test_product_cells_two_dimensions_yields_the_full_cross_product() -> None:
    dims = {
        "selection": [("elite", {}), ("roulette", {})],
        "crossover": [("one_point", {}), ("uniform", {})],
    }
    cells = list(product_cells(dims))
    # A genuine 2x2 cross product, proving the mechanism is itertools.product
    # even though this plan's own shipped arms each use exactly one dimension.
    assert len(cells) == 4
    ids = {cell.cell_id for cell in cells}
    assert ids == {
        "selection-elite-crossover-one_point",
        "selection-elite-crossover-uniform",
        "selection-roulette-crossover-one_point",
        "selection-roulette-crossover-uniform",
    }


def test_product_cells_merges_overrides_with_later_dimensions_winning() -> None:
    dims = {
        "a": [("x", {"population": 10, "children": 5})],
        "b": [("y", {"children": 99})],
    }
    (cell,) = list(product_cells(dims))
    assert cell.overrides == {"population": 10, "children": 99}


# --- apply_overrides -----------------------------------------------------------


def test_apply_overrides_whole_value_replacement() -> None:
    baseline = {"population": 30, "parents": {"method": "elite"}, "children": 40}
    result = apply_overrides(baseline, {"parents": {"method": "roulette"}})
    assert result["parents"] == {"method": "roulette"}
    assert result["population"] == 30
    assert result["children"] == 40
    # The baseline dict itself is never mutated.
    assert baseline["parents"] == {"method": "elite"}


def test_apply_overrides_resolves_children_ratio_via_round() -> None:
    baseline = {"population": 30, "children": 40}
    result = apply_overrides(baseline, {"children_ratio": 2.0})
    assert result["children"] == round(2.0 * 30)
    assert "children_ratio" not in result


@pytest.mark.parametrize("ratio, population, expected", [(0.5, 30, 15), (1.0, 30, 30), (2.0, 30, 60)])
def test_apply_overrides_children_ratio_matches_expected_rounding(ratio, population, expected) -> None:
    baseline = {"population": population}
    result = apply_overrides(baseline, {"children_ratio": ratio})
    assert result["children"] == expected


# --- load_matrix_spec validation ------------------------------------------------


def _write_spec(tmp_path: Path, **overrides) -> Path:
    import json

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"population": 10, "children": 10}), encoding="utf-8")

    spec = {
        "baseline": "baseline.json",
        "base_seed": 1,
        "seeds": 2,
        "out_root": "runs/_test",
        "image": "image.png",
        "canvas": 32,
        "triangles": 4,
        "arms": {"selection": {"elite": {}}},
    }
    spec.update(overrides)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path


def test_load_matrix_spec_rejects_seeds_zero(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path, seeds=0)
    with pytest.raises(MatrixSpecError, match="seeds"):
        load_matrix_spec(spec_path)


def test_load_matrix_spec_rejects_seeds_non_integer(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path, seeds=1.5)
    with pytest.raises(MatrixSpecError, match="seeds"):
        load_matrix_spec(spec_path)


def test_load_matrix_spec_rejects_empty_arms(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path, arms={})
    with pytest.raises(MatrixSpecError, match="arms"):
        load_matrix_spec(spec_path)


def test_load_matrix_spec_resolves_baseline_relative_to_spec_dir(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    spec = load_matrix_spec(spec_path)
    assert spec.baseline_path == (tmp_path / "baseline.json").resolve()
    assert spec.seeds == 2
