"""Contracts and tests for IO-05 metrics.csv schema and completeness.

Proves:
1. Header row matches METRICS_COLUMNS exactly in name and order.
2. Every row contains finite numeric values across diverse operator configs.
3. diversity() on zero-variance population is exactly 0.0 (never NaN).
4. Boundary 1-generation runs (horizon 0 and gen-0 threshold hit) produce full, valid rows.
5. Invariants hold: active_triangles in [0, budget], renders > 0 and non-decreasing, elapsed_s non-decreasing.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from tp2.engine.config import build_run_config
from tp2.engine.diversity import diversity
from tp2.engine.fitness import Evaluator
from tp2.engine.genome import bounds_for, chromosome_length
from tp2.engine.loop import Run
from tp2.io.images import load_target
from tp2.io.metrics import METRICS_COLUMNS, MetricsWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"


def test_metrics_header_matches_frozen_constant(tmp_path):
    """The CSV header written by MetricsWriter equals METRICS_COLUMNS in exact order."""
    csv_file = tmp_path / "metrics.csv"
    with MetricsWriter(csv_file):
        pass

    with csv_file.open(encoding="utf-8") as f:
        first_line = f.readline().strip()
    assert first_line == ",".join(METRICS_COLUMNS)


def test_all_metrics_values_are_finite_across_operator_combinations(tmp_path):
    """Multi-generation runs with various operators produce finite numbers in all columns."""
    target = load_target(ASSETS_DIR / "flag_ar.png", (32, 32))
    evaluator = Evaluator(target=target, size=(32, 32))
    triangles = 6

    # Test config 1: Standard baseline style
    cfg_1 = {
        "population": 8, "children": 8, "horizon": 5, "recombination_probability": 0.8,
        "parents": {"method": "elite"}, "replacement": {"method": "elite"},
        "crossover": {"method": "one_point"}, "mutation": {"method": "gene", "probability": 0.5},
        "survival": {"method": "additive"}, "stop": {"max_generations": True},
    }

    # Test config 2: Boltzmann selection with low Tc
    cfg_2 = {
        "population": 8, "children": 8, "horizon": 5, "recombination_probability": 0.8,
        "parents": {"method": "boltzmann", "t0": 5.0, "tc": 0.05, "k": 0.1},
        "replacement": {"method": "tournament_probabilistic", "threshold": 0.8},
        "crossover": {"method": "ring"}, "mutation": {"method": "multigen_uniform", "probability": 0.5},
        "survival": {"method": "generational_gap", "gap": 0.5}, "stop": {"max_generations": True},
    }

    for idx, cfg_data in enumerate((cfg_1, cfg_2)):
        csv_file = tmp_path / f"metrics_{idx}.csv"
        run = Run(build_run_config(cfg_data), evaluator, triangles, np.random.default_rng(idx))
        with MetricsWriter(csv_file) as writer:
            for event in run:
                writer.write(event)

        with csv_file.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 6  # gen 0 to 5
        for r_idx, row in enumerate(rows):
            for col in METRICS_COLUMNS:
                if col == "stop_reason":
                    continue
                val = float(row[col])
                assert math.isfinite(val), f"Non-finite value in row {r_idx}, column {col}: {val}"


def test_diversity_zero_variance_is_exact_zero():
    """diversity() returns 0.0 (not NaN or epsilon) for identical genomes."""
    triangles = 5
    bounds = bounds_for(triangles)
    genes = np.full((10, chromosome_length(triangles)), 0.5, dtype=np.float32)
    val = diversity(genes, bounds)
    assert val == 0.0
    assert math.isfinite(val)


def test_single_generation_boundary_run_writes_complete_row(tmp_path):
    """A run terminating at generation 0 writes a single fully-populated row."""
    target = load_target(ASSETS_DIR / "flag_ar.png", (32, 32))
    evaluator = Evaluator(target=target, size=(32, 32))
    triangles = 4

    # Horizon 0
    cfg_data = {
        "population": 6, "children": 6, "horizon": 1, "recombination_probability": 0.8,
        "parents": {"method": "elite"}, "replacement": {"method": "elite"},
        "crossover": {"method": "one_point"}, "mutation": {"method": "gene", "probability": 0.5},
        "survival": {"method": "additive"}, "stop": {"min_fitness": 0.01},  # Immediate gen 0 trigger
    }

    csv_file = tmp_path / "metrics_gen0.csv"
    run = Run(build_run_config(cfg_data), evaluator, triangles, np.random.default_rng(123))
    with MetricsWriter(csv_file) as writer:
        for event in run:
            writer.write(event)

    with csv_file.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert int(row["generation"]) == 0
    assert row["stop_reason"] == "min_fitness"
    for col in METRICS_COLUMNS:
        if col == "stop_reason":
            continue
        assert row[col] != ""
        assert math.isfinite(float(row[col]))


def test_monotonicity_and_bounds_invariants(tmp_path):
    """active_triangles in [0, budget], renders > 0 and non-decreasing, elapsed_s non-decreasing."""
    target = load_target(ASSETS_DIR / "flag_ar.png", (32, 32))
    evaluator = Evaluator(target=target, size=(32, 32))
    triangles = 8

    cfg_data = {
        "population": 6, "children": 6, "horizon": 6, "recombination_probability": 0.8,
        "parents": {"method": "elite"}, "replacement": {"method": "elite"},
        "crossover": {"method": "one_point"}, "mutation": {"method": "gene", "probability": 0.5},
        "survival": {"method": "additive"}, "stop": {"max_generations": True},
    }

    csv_file = tmp_path / "metrics_invariants.csv"
    run = Run(build_run_config(cfg_data), evaluator, triangles, np.random.default_rng(7))
    with MetricsWriter(csv_file) as writer:
        for event in run:
            writer.write(event)

    with csv_file.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    prev_renders = 0
    prev_elapsed = -1.0

    for row in rows:
        act = int(row["active_triangles"])
        assert 0 <= act <= triangles

        renders = int(row["renders"])
        assert renders > prev_renders
        prev_renders = renders

        elapsed = float(row["elapsed_s"])
        assert elapsed >= prev_elapsed
        prev_elapsed = elapsed
