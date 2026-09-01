"""Config schema, range, and desugaring contracts.

Scope note: `build_run_config` takes a single already-merged dict (no
`overrides` parameter) and `RunConfig` carries no `canvas`/`triangle_budget`
fields -- both are CLI-only concerns validated in `tp2/cli.py`, not in this
module. That is inherited 02-01 architecture (see 02-01-SUMMARY.md gap #6),
not a gap this plan's own must-haves ask it to close, so no CLI-override-merge
test or canvas/triangle_budget range test is written here.
"""

from __future__ import annotations

import json

import pytest

from tp2.engine.config import SELECTION_SLOTS, ConfigError, build_run_config, desugar_selection, load_config
from tp2.engine.operators.registry import BLEND_MAX_DEPTH


def _config(**overrides) -> dict:
    base = {
        "population": 4, "children": 4, "recombination_probability": 0.8,
        "parents": {"method": "elite"}, "replacement": {"method": "elite"},
        "crossover": {"method": "one_point", "boundary": "triangle"},
        "mutation": {"method": "gene", "probability": 0.5},
        "survival": {"method": "additive"}, "stop": {"max_generations": 2},
    }
    base.update(overrides)
    return base


# --- unknown top-level keys ---------------------------------------------------


def test_unknown_top_level_key_is_rejected_naming_the_key() -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_run_config(_config(bogus_key=123))
    assert "bogus_key" in str(excinfo.value)


# --- numeric range checks -----------------------------------------------------


@pytest.mark.parametrize("value", [0, -1])
def test_population_below_minimum_is_rejected_naming_the_key(value) -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_run_config(_config(population=value))
    assert "population" in str(excinfo.value)


@pytest.mark.parametrize("value", [0, -1])
def test_children_below_minimum_is_rejected_naming_the_key(value) -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_run_config(_config(children=value))
    assert "children" in str(excinfo.value)


def test_odd_children_count_is_accepted() -> None:
    config = build_run_config(_config(children=5))
    assert config.children == 5


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_recombination_probability_out_of_range_is_rejected_naming_the_key(value) -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_run_config(_config(recombination_probability=value))
    assert "recombination_probability" in str(excinfo.value)


@pytest.mark.parametrize("value", [5.0, True, -1, "5"])
def test_max_generations_type_contract_names_the_key(value) -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_run_config(_config(stop={"max_generations": value}))
    assert "max_generations" in str(excinfo.value)


# --- crossover boundary --------------------------------------------------------


def test_unknown_crossover_boundary_is_rejected_naming_key_value_and_accepted_set() -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_run_config(_config(crossover={"method": "one_point", "boundary": "bogus"}))
    message = str(excinfo.value)
    assert "boundary" in message and "bogus" in message
    assert "gene" in message and "triangle" in message


# --- blend validation -----------------------------------------------------------


def test_blend_coefficient_out_of_range_is_rejected_naming_the_key() -> None:
    spec = {"method": "blend", "coefficient": 1.5, "method_1": {"method": "elite"}, "method_2": {"method": "random"}}
    with pytest.raises(ConfigError) as excinfo:
        build_run_config(_config(parents=spec))
    assert "coefficient" in str(excinfo.value)


def test_blend_missing_method_2_is_rejected_naming_the_key() -> None:
    spec = {"method": "blend", "coefficient": 0.5, "method_1": {"method": "elite"}}
    with pytest.raises(ConfigError) as excinfo:
        build_run_config(_config(parents=spec))
    assert "method_2" in str(excinfo.value)


def _chain_blend(depth: int) -> dict:
    """A blend spec nested `depth` levels deep, terminating in a leaf `elite`."""
    spec: dict = {"method": "elite"}
    for _ in range(depth):
        spec = {"method": "blend", "coefficient": 0.5, "method_1": spec, "method_2": {"method": "random"}}
    return spec


def test_blend_nested_at_the_cap_depth_builds() -> None:
    config = build_run_config(_config(parents=_chain_blend(BLEND_MAX_DEPTH)))
    assert config.parents is not None


def test_blend_nested_deeper_than_the_cap_is_rejected_naming_the_cap() -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_run_config(_config(parents=_chain_blend(BLEND_MAX_DEPTH + 1)))
    assert str(BLEND_MAX_DEPTH) in str(excinfo.value)


def test_blend_spec_builds_identically_in_the_replacement_slot() -> None:
    spec = {"method": "blend", "coefficient": 0.5, "method_1": {"method": "elite"}, "method_2": {"method": "random"}}
    config = build_run_config(_config(replacement=spec))
    assert config.replacement is not None


# --- desugaring / archived effective config -------------------------------------


def test_desugar_selection_is_currently_the_identity() -> None:
    spec = {"method": "blend", "coefficient": 0.4, "method_1": {"method": "elite"}, "method_2": {"method": "random"}}
    assert desugar_selection(spec) == spec


def test_selection_slots_names_parents_and_replacement() -> None:
    assert set(SELECTION_SLOTS) == {"parents", "replacement"}


def test_effective_config_records_the_desugared_blend_spec_not_a_shorthand() -> None:
    blend_spec = {
        "method": "blend", "coefficient": 0.6,
        "method_1": {"method": "elite"}, "method_2": {"method": "random"},
    }
    config = build_run_config(_config(parents=json.loads(json.dumps(blend_spec))))
    assert config.effective["parents"] == blend_spec
    assert config.effective["parents"]["coefficient"] == 0.6
    assert config.effective["parents"]["method_1"] == {"method": "elite"}
    assert config.effective["parents"]["method_2"] == {"method": "random"}


def test_load_config_reads_a_json_file_end_to_end(tmp_path) -> None:
    """The one test that goes through `load_config` so the file path is
    exercised too, not just `build_run_config` on an in-memory dict."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    config = build_run_config(load_config(path))
    assert config.population == 4
