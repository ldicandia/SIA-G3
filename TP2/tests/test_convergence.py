"""Multi-seed convergence gate for the shipped default configuration (ROADMAP Success Criterion 3).

Proves the engine actually converges on a real target (assets/flag_ar.png)
above fitness 0.97 on seeds 1, 7, and 42 within its configured generation cap,
with non-decreasing best fitness and exact render accounting.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tp2.cli import main
from tp2.engine.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.json"
TARGET_IMAGE_PATH = PROJECT_ROOT / "assets" / "flag_ar.png"
SEEDS = (1, 7, 42)
TRIANGLES = 30
FITNESS_THRESHOLD = 0.97


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_baseline_convergence_flag_target(seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run shipped configs/baseline.json against flag_ar.png and assert > 0.97 convergence."""
    # Ensure what is tested is the shipped config file
    raw_cfg = load_config(BASELINE_CONFIG_PATH)
    child_count = raw_cfg["children"]
    max_generations = raw_cfg["stop"]["max_generations"]

    out_dir = tmp_path / f"run_seed_{seed}"
    args = [
        "tp2",
        "--image", str(TARGET_IMAGE_PATH),
        "--triangles", str(TRIANGLES),
        "--config", str(BASELINE_CONFIG_PATH),
        "--seed", str(seed),
        "--out", str(out_dir),
    ]
    monkeypatch.setattr("sys.argv", args)
    main()

    metrics_file = out_dir / "metrics.csv"
    assert metrics_file.exists(), f"metrics.csv was not written for seed {seed}"

    with metrics_file.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == max_generations + 1, f"Expected {max_generations + 1} rows, got {len(rows)}"

    fitnesses = [float(r["best_fitness"]) for r in rows]
    renders = [int(r["renders"]) for r in rows]

    # Monotonically non-decreasing best fitness
    for i in range(len(fitnesses) - 1):
        assert fitnesses[i + 1] >= fitnesses[i], (
            f"Seed {seed}: best_fitness decreased at gen {i + 1} ({fitnesses[i]} -> {fitnesses[i + 1]})"
        )

    # Render counter rises by exactly child count per generation
    for i in range(len(renders) - 1):
        step = renders[i + 1] - renders[i]
        assert step == child_count, (
            f"Seed {seed}: render step at gen {i + 1} was {step}, expected child count {child_count}"
        )

    last_row = rows[-1]
    assert last_row["stop_reason"] == "max_generations", (
        f"Seed {seed}: unexpected stop reason {last_row['stop_reason']}"
    )

    final_fitness = fitnesses[-1]
    assert final_fitness > FITNESS_THRESHOLD, (
        f"Seed {seed}: final fitness {final_fitness:.5f} did not exceed threshold {FITNESS_THRESHOLD}"
    )
