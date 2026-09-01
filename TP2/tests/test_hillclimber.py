"""Import-boundary lock, render-count honesty, and the mislabeling
prohibition for `tp2/baselines/hillclimber.py` (EXP-05).

Fixtures `flat_target`, `target_image`, `project_root`, and `read_json` come
from `tests/conftest.py`.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tp2.baselines import hillclimber
from tp2.baselines.hillclimber import HillclimbResult, run_hillclimb
from tp2.engine.fitness import Evaluator
from tp2.engine.operators.registry import MUTATION
from tp2.engine.stop import build_stop_set

HILLCLIMBER_PATH = Path(hillclimber.__file__)


# --- import boundary ---------------------------------------------------------


def test_hillclimber_does_not_import_selection_crossover_or_survival() -> None:
    source = HILLCLIMBER_PATH.read_text(encoding="utf-8")
    for banned in (
        "from tp2.engine.operators.selection",
        "from tp2.engine.operators.crossover",
        "from tp2.engine.operators.survival",
    ):
        assert banned not in source, f"{banned!r} must never appear in hillclimber.py"


# --- mislabeling prohibition (judgment-tier; best-effort mechanical proxy) --


def test_module_docstring_never_uses_bare_ga() -> None:
    docstring = hillclimber.__doc__ or ""
    assert not re.search(r"\bga\b", docstring, re.IGNORECASE), (
        "the module's own top-level docstring must never self-label this algorithm with the bare word 'GA'"
    )


def test_source_and_help_never_use_baseline_ga_phrasing() -> None:
    source = HILLCLIMBER_PATH.read_text(encoding="utf-8").lower()
    help_text = hillclimber.build_parser().format_help().lower()
    combined = source + "\n" + help_text
    assert "baseline ga" not in combined
    assert "ga baseline" not in combined


# --- render-count honesty ----------------------------------------------------


@pytest.mark.parametrize("horizon", [3, 10])
def test_renders_equal_one_plus_iterations(flat_target, horizon) -> None:
    evaluator = Evaluator(flat_target, (8, 8))
    mutate = MUTATION.build({"method": "gene", "probability": 1.0})
    stop_set = build_stop_set({"max_generations": True}, horizon)
    rng = np.random.default_rng(1)

    result = run_hillclimb(evaluator, mutate, 2, rng, stop_set)

    assert result.iterations == horizon
    assert evaluator.renders == 1 + horizon
    assert evaluator.renders == 1 + result.iterations
    assert result.stop_reason == "max_generations"


def test_best_fitness_never_decreases_and_a_rejected_mutant_never_changes_the_incumbent(flat_target) -> None:
    evaluator = Evaluator(flat_target, (8, 8))
    mutate = MUTATION.build({"method": "gene", "probability": 1.0})
    stop_set = build_stop_set({"max_generations": True}, 40)
    rng = np.random.default_rng(5)
    events: list = []

    result = run_hillclimb(evaluator, mutate, 3, rng, stop_set, observer=events.append)

    assert len(events) == 41  # generation 0's initial event + 40 iterations
    fitnesses = [event.best_fitness for event in events]
    assert fitnesses == sorted(fitnesses), "best_fitness must never decrease under strict accept-if-better"

    genomes = [event.best_genes.tobytes() for event in events]
    for prev_fitness, cur_fitness, prev_genes, cur_genes in zip(fitnesses, fitnesses[1:], genomes, genomes[1:]):
        if cur_fitness == prev_fitness:
            assert cur_genes == prev_genes, "fitness unchanged but genome differs -- a rejected mutant leaked in"

    assert isinstance(result, HillclimbResult)
    assert result.stop_reason == "max_generations"


def test_diversity_is_always_zero_and_the_population_has_no_spread(flat_target) -> None:
    evaluator = Evaluator(flat_target, (8, 8))
    mutate = MUTATION.build({"method": "gene", "probability": 1.0})
    stop_set = build_stop_set({"max_generations": True}, 5)
    events: list = []

    run_hillclimb(evaluator, mutate, 2, np.random.default_rng(2), stop_set, observer=events.append)

    assert all(event.diversity == 0.0 for event in events)
    assert all(event.mean_fitness == event.worst_fitness == event.best_fitness for event in events)


# --- determinism --------------------------------------------------------------


def _rows_excluding_elapsed(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    elapsed_index = rows[0].index("elapsed_s")
    return [[cell for index, cell in enumerate(row) if index != elapsed_index] for row in rows]


def test_two_runs_same_seed_produce_byte_identical_best_png(tmp_path, target_image, project_root) -> None:
    baseline_config = project_root / "configs" / "baseline.json"
    common = [
        sys.executable, "-m", "tp2.baselines.hillclimber",
        "--image", str(target_image),
        "--triangles", "6",
        "--canvas", "16",
        "--config", str(baseline_config),
        "--seed", "3",
        "--max-renders", "12",
        "--allow-outside",
    ]
    out_a, out_b = tmp_path / "hc_a", tmp_path / "hc_b"
    subprocess.run([*common, "--out", str(out_a)], cwd=project_root, check=True)
    subprocess.run([*common, "--out", str(out_b)], cwd=project_root, check=True)

    assert (out_a / "best.png").read_bytes() == (out_b / "best.png").read_bytes()
    assert _rows_excluding_elapsed(out_a / "metrics.csv") == _rows_excluding_elapsed(out_b / "metrics.csv")


# --- artifact contract --------------------------------------------------------


def test_run_json_records_the_algorithm_label(tmp_path, target_image, project_root, read_json) -> None:
    baseline_config = project_root / "configs" / "baseline.json"
    out = tmp_path / "hc_label"
    argv = [
        "--image", str(target_image),
        "--triangles", "6",
        "--canvas", "16",
        "--config", str(baseline_config),
        "--seed", "9",
        "--max-renders", "8",
        "--out", str(out),
        "--allow-outside",
    ]

    assert hillclimber.main(argv) == 0

    payload = read_json(out / "run.json")
    assert payload["algorithm"] == "hillclimb_1p1"
    assert payload["stop_reason"] == "max_generations"
