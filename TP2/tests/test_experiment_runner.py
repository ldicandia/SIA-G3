"""Edge cases for the matrix runner (EXP-01/EXP-02): a seeds=1 cell, seed and
path uniqueness at the real 75-job scale (computed without running any of
them), the silent-failure guard, and ordering-independence of cell identity.

Fixtures `project_root` comes from `tests/conftest.py`.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from tp2.engine.events import GenerationEvent
from tp2.experiments import runner
from tp2.experiments.matrix import MatrixSpec, load_matrix_spec
from tp2.experiments.runner import build_jobs, run_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_MATRIX_PATH = PROJECT_ROOT / "configs" / "experiments" / "main_matrix.json"

# Tracer-scale overrides shared by every in-process test in this file, so a
# real Evaluator/Run round-trip stays fast.
_TRACER_SCALE = {"population": 6, "children": 6, "horizon": 5, "stop": {"max_generations": True}}


@pytest.fixture
def matrix_out_root(project_root: Path):
    """A fresh, project-relative output root -- prepare_run_dir refuses paths
    outside the project by default, so test output must live under
    `runs/`, not pytest's own tmp_path."""
    root = project_root / "runs" / f"_pytest_matrix_{uuid.uuid4().hex[:8]}"
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _tracer_spec(out_root: Path, seeds: int, arms: dict, project_root: Path) -> MatrixSpec:
    return MatrixSpec(
        baseline_path=project_root / "configs" / "baseline.json",
        base_seed=999,
        seeds=seeds,
        arms=arms,
        out_root=out_root,
        image_path=project_root / "assets" / "flag_ar.png",
        canvas=32,
        triangles=6,
        scale_overrides=dict(_TRACER_SCALE),
    )


# --- seeds=1 --------------------------------------------------------------


def test_seeds_one_runs_one_replicate_per_cell_without_error(matrix_out_root, project_root) -> None:
    spec = _tracer_spec(
        matrix_out_root,
        seeds=1,
        arms={
            "selection": {
                "elite": {},
                "roulette": {"parents": {"method": "roulette"}, "replacement": {"method": "roulette"}},
            }
        },
        project_root=project_root,
    )
    results = run_matrix(spec, jobs=1)
    assert len(results) == 2  # 2 cells x 1 seed -- no code path assumes seeds >= 2
    assert all(r.ok for r in results), results

    for cell_id in ("selection-elite", "selection-roulette"):
        run_dir = matrix_out_root / cell_id / "seed0"
        assert (run_dir / "best.png").exists(), run_dir
        assert (run_dir / "metrics.csv").exists(), run_dir
        assert (run_dir / "triangles.json").exists(), run_dir
        assert (run_dir / "run.json").exists(), run_dir


# --- seed and path uniqueness at the real 75-job scale (no execution) ------


def test_derive_seed_has_no_collision_over_the_full_75_job_space() -> None:
    spec = load_matrix_spec(MAIN_MATRIX_PATH)
    jobs = build_jobs(spec)
    assert len(jobs) == 75  # 15 cells x 5 seeds
    seeds = {job.seed for job in jobs}
    assert len(seeds) == 75, "derive_seed collided somewhere across the 75-job space"


def test_planned_output_paths_are_all_unique_over_the_full_75_job_space() -> None:
    spec = load_matrix_spec(MAIN_MATRIX_PATH)
    jobs = build_jobs(spec)
    assert len(jobs) == 75
    paths = {str(job.out_dir) for job in jobs}
    assert len(paths) == 75, "a planned output path collided somewhere across the 75-job space"


# --- silent-failure guard ---------------------------------------------------


def test_one_failing_cell_completes_every_other_cell_and_reports_the_failure(
    tmp_path: Path, matrix_out_root: Path, project_root: Path
) -> None:
    spec_data = {
        "baseline": str(project_root / "configs" / "baseline.json"),
        "base_seed": 42,
        "seeds": 1,
        "out_root": str(matrix_out_root),
        "image": str(project_root / "assets" / "flag_ar.png"),
        "canvas": 32,
        "triangles": 6,
        "scale_overrides": _TRACER_SCALE,
        "arms": {
            "selection": {
                "good": {},
                "bad": {"parents": {"method": "not_a_real_method"}},
            }
        },
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_data), encoding="utf-8")

    exit_code = runner.main(["--spec", str(spec_path), "--out", str(matrix_out_root), "--jobs", "1"])
    assert exit_code != 0

    good_dir = matrix_out_root / "selection-good" / "seed0"
    assert (good_dir / "best.png").exists()
    assert (good_dir / "metrics.csv").exists()
    assert (good_dir / "triangles.json").exists()
    assert (good_dir / "run.json").exists()

    bad_dir = matrix_out_root / "selection-bad" / "seed0"
    assert not (bad_dir / "best.png").exists(), "a failed cell must never write a partial artifact set"

    failures_path = matrix_out_root / "FAILURES.json"
    assert failures_path.exists()
    failures = json.loads(failures_path.read_text(encoding="utf-8"))
    assert len(failures) == 1
    assert failures[0]["cell_id"] == "selection-bad"
    assert failures[0]["replicate_index"] == 0
    assert "not_a_real_method" in failures[0]["error"]


# --- CR-01: a mid-run crash must never leave a file named metrics.csv -------


def test_a_crash_mid_run_leaves_no_file_named_metrics_csv(
    monkeypatch, matrix_out_root: Path, project_root: Path
) -> None:
    """`MetricsWriter` flushes after every row, so without the atomic-rename
    fix a mid-loop crash leaves a syntactically valid but truncated
    `metrics.csv` behind -- indistinguishable from a complete replicate to
    `tp2/experiments/aggregate.py`'s `load_seed_curves`, which just globs
    `seed*/metrics.csv`. `run_cell_seed` must write under a temporary name
    and only rename it to `metrics.csv` once the entire run (including
    `run.json`) has succeeded, so a crashed seed's directory has no file
    literally named `metrics.csv` at all."""
    spec = _tracer_spec(
        matrix_out_root, seeds=1, arms={"selection": {"elite": {}}}, project_root=project_root,
    )
    job = build_jobs(spec)[0]

    class _CrashingRun:
        """Stands in for `tp2.engine.loop.Run`: yields one real-shaped event
        then raises, simulating a crash several generations into a run --
        after `MetricsWriter` has already flushed at least one row."""

        def __init__(self, *_args, **_kwargs) -> None:
            self.result = None

        def __iter__(self):
            genes = np.zeros((6, 11), dtype=np.float32)
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            yield GenerationEvent(0, 10, 0.01, 0.5, 0.5, 0.4, 0.3, 0.1, 3, genes, frame, "")
            raise RuntimeError("synthetic mid-run crash")

    monkeypatch.setattr(runner, "Run", _CrashingRun)

    summary = runner.run_cell_seed(job)

    assert summary.ok is False
    assert "synthetic mid-run crash" in summary.error

    run_dir = job.out_dir
    assert not (run_dir / "metrics.csv").exists(), "a crashed run must never leave a file named metrics.csv"
    assert (run_dir / "metrics.csv.tmp").exists(), "the in-progress write is still on disk under its temp name"
    assert not (run_dir / "run.json").exists()


# --- ordering independence --------------------------------------------------


def _rebuilt_with(spec: MatrixSpec, **overrides) -> MatrixSpec:
    fields = dict(
        baseline_path=spec.baseline_path, base_seed=spec.base_seed, seeds=spec.seeds,
        arms=spec.arms, out_root=spec.out_root, image_path=spec.image_path,
        canvas=spec.canvas, triangles=spec.triangles, scale_overrides=spec.scale_overrides,
    )
    fields.update(overrides)
    return MatrixSpec(**fields)


def test_reordering_the_arms_mapping_preserves_cell_id_set_and_per_cell_seeds() -> None:
    spec = load_matrix_spec(MAIN_MATRIX_PATH)
    jobs_normal = build_jobs(spec)

    # Rebuilt in Python with keys inserted in reverse order -- JSON key order
    # is not semantically meaningful, and this proves it structurally rather
    # than by hand-editing a second spec file.
    reversed_arms = dict(reversed(list(spec.arms.items())))
    jobs_reversed = build_jobs(_rebuilt_with(spec, arms=reversed_arms))

    ids_normal = {job.cell_id for job in jobs_normal}
    ids_reversed = {job.cell_id for job in jobs_reversed}
    assert ids_normal == ids_reversed

    # cell_index is derived from a STABLE SORT of cell_id, never raw list
    # position -- so the seed for a given (cell_id, replicate_index) is
    # identical no matter which order the arms dict was iterated in to build
    # the cell list.
    seeds_normal = {(job.cell_id, job.replicate_index): job.seed for job in jobs_normal}
    seeds_reversed = {(job.cell_id, job.replicate_index): job.seed for job in jobs_reversed}
    assert seeds_normal == seeds_reversed


def test_appending_a_new_arm_at_the_end_never_changes_existing_cells_seeds() -> None:
    spec = load_matrix_spec(MAIN_MATRIX_PATH)
    seeds_before = {(job.cell_id, job.replicate_index): job.seed for job in build_jobs(spec)}

    extended_arms = dict(spec.arms)
    extended_arms["extra_arm"] = {"only-label": {}}
    seeds_after = {
        (job.cell_id, job.replicate_index): job.seed
        for job in build_jobs(_rebuilt_with(spec, arms=extended_arms))
    }

    for key, seed in seeds_before.items():
        assert seeds_after[key] == seed, f"appending a new arm changed the derived seed for {key}"
