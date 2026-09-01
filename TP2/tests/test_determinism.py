"""Per-seed reproducibility: the claim every Phase 4 experiment rests on.

Determinism is asserted on the *file bytes* of `best.png` and `triangles.json`,
never on an in-memory array. A run that scores one frame and writes another
would satisfy an in-memory comparison and fail these (threat T-01-07).
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from tp2.cli import resolve_seed


def test_same_seed_is_byte_identical_across_different_out_directories(tmp_path, run_slice0) -> None:
    first = run_slice0(tmp_path / "a", seed=7)
    second = run_slice0(tmp_path / "b", seed=7)

    assert (first / "best.png").read_bytes() == (second / "best.png").read_bytes()
    assert (first / "triangles.json").read_bytes() == (second / "triangles.json").read_bytes()


def test_a_different_seed_produces_a_different_image(tmp_path, run_slice0) -> None:
    seven = run_slice0(tmp_path / "seed7", seed=7)
    eight = run_slice0(tmp_path / "seed8", seed=8)

    assert (seven / "best.png").read_bytes() != (eight / "best.png").read_bytes()


def test_seed_zero_is_a_literal_seed_and_never_treated_as_absent(tmp_path, run_slice0, read_json) -> None:
    # `--seed 0` is falsy; a truthiness check in seed resolution would silently
    # draw a random seed here and the run would not be reproducible.
    assert resolve_seed(argparse.Namespace(seed=0)) == 0

    first = run_slice0(tmp_path / "zero_a", seed=0)
    assert read_json(first / "run.json")["seed"] == 0

    second = run_slice0(tmp_path / "zero_b", seed=0)
    assert (first / "best.png").read_bytes() == (second / "best.png").read_bytes()


def test_an_omitted_seed_is_drawn_archived_and_reproducible_afterwards(tmp_path, run_slice0, read_json) -> None:
    drawn_run = run_slice0(tmp_path / "drawn", seed=None)
    archived = read_json(drawn_run / "run.json")["seed"]

    assert isinstance(archived, int) and not isinstance(archived, bool)

    replay = run_slice0(tmp_path / "replay", seed=archived)
    assert (drawn_run / "best.png").read_bytes() == (replay / "best.png").read_bytes()
    assert (drawn_run / "triangles.json").read_bytes() == (replay / "triangles.json").read_bytes()


def test_an_omitted_seed_is_actually_drawn_rather_than_defaulted(tmp_path, run_slice0, read_json) -> None:
    seeds = {
        read_json(run_slice0(tmp_path / f"draw{index}", seed=None) / "run.json")["seed"]
        for index in range(3)
    }
    assert len(seeds) == 3


# --- single source of randomness -------------------------------------------
#
# Checked by walking the AST rather than by grepping the text, so a mention
# inside a docstring cannot fail the check and a real call cannot hide behind
# unusual formatting.


def _dotted_name(func: ast.expr) -> str:
    parts: list[str] = []
    node: ast.expr = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _called_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [_dotted_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _engine_sources(project_root: Path) -> list[Path]:
    sources = sorted(project_root.joinpath("tp2").rglob("*.py"))
    assert sources, "no modules found under tp2/"
    return sources


def test_no_module_under_tp2_seeds_a_global_generator(project_root) -> None:
    offenders = [
        (path.relative_to(project_root).as_posix(), name)
        for path in _engine_sources(project_root)
        for name in _called_names(path)
        if name.split(".")[-2:] == ["random", "seed"]
    ]
    assert offenders == [], f"global seeding leaks across runs: {offenders}"


def test_the_generator_is_constructed_exactly_once_per_composition_root(project_root) -> None:
    # 04-02: `tp2/baselines/hillclimber.py` is a second, deliberate
    # composition root -- `python -m tp2.baselines.hillclimber` runs
    # standalone, without going through `tp2/cli.py`, so it needs its own
    # seeded Generator.
    # 04-03: `tp2/experiments/runner.py` is a third, deliberate composition
    # root -- each `multiprocessing.Pool` worker process (`run_cell_seed`)
    # independently builds its own `np.random.default_rng(job.seed)` from a
    # per-cell-per-replicate DERIVED seed (never inherited process state),
    # matching this exact codebase's single-injected-rng convention rather
    # than the never-built `tp2/engine/rng.py` per-family stream split. The
    # invariant this test protects is unchanged (exactly one construction per
    # composition root, never inside the engine or an operator), just widened
    # to name all three roots explicitly.
    constructions = {
        path.relative_to(project_root).as_posix(): sum(
            1 for name in _called_names(path) if name.split(".")[-1] == "default_rng"
        )
        for path in _engine_sources(project_root)
    }
    building = {path: count for path, count in constructions.items() if count}

    expected = {"tp2/cli.py": 1, "tp2/baselines/hillclimber.py": 1, "tp2/experiments/runner.py": 1}
    assert building == expected, (
        "exactly one numpy.random.Generator must be created per composition root "
        f"({', '.join(sorted(expected))}), got {building}"
    )

