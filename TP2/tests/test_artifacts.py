"""Run-directory contracts: the four artifacts, their schemas, and the guards.

A run directory is evidence. These tests pin what it contains, and pin that it
is never silently overwritten, merged into, or written outside the project.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from tp2 import cli
from tp2.engine.genome import ACTIVE, GENES_PER_TRIANGLE, active_count, random_population
from tp2.io.artifacts import RunDirError, prepare_run_dir, write_triangles_json
from tp2.io.metrics import METRICS_COLUMNS

ARTIFACTS = ("best.png", "triangles.json", "run.json", "metrics.csv")


def _rows(run_dir):
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


# --- the four artifacts -----------------------------------------------------


def test_a_run_writes_all_four_artifacts(tmp_path, run_slice0) -> None:
    run_dir = run_slice0(tmp_path / "run")
    assert sorted(path.name for path in run_dir.iterdir()) == sorted(ARTIFACTS)


def test_triangles_json_enumerates_every_triangle(tmp_path, run_slice0, read_json) -> None:
    payload = read_json(run_slice0(tmp_path / "run", triangles=8, canvas=32) / "triangles.json")

    assert payload["canvas"] == {"width": 32, "height": 32}
    assert payload["budget"] == 8
    assert len(payload["triangles"]) == 8
    assert [entry["index"] for entry in payload["triangles"]] == list(range(8))

    for entry in payload["triangles"]:
        assert len(entry["points"]) == 3
        assert all(len(point) == 2 and all(isinstance(value, int) for value in point) for point in entry["points"])
        assert len(entry["color"]) == 3
        assert all(isinstance(channel, int) and 0 <= channel <= 255 for channel in entry["color"])
        assert isinstance(entry["alpha"], int) and 0 <= entry["alpha"] <= 255
        assert isinstance(entry["active"], bool)

    assert payload["active_count"] == sum(entry["active"] for entry in payload["triangles"])


def test_triangles_json_active_count_matches_the_genome(tmp_path, read_json) -> None:
    # Driven off a known chromosome rather than a run so the expected count is
    # independent of which individual happened to win.
    genes = random_population(np.random.default_rng(11), 1, 6)[0]
    table = genes.reshape(-1, GENES_PER_TRIANGLE)
    table[:, ACTIVE] = [0.5, 0.4999, 0.9, 0.0, 1.0, 0.49]

    destination = tmp_path / "triangles.json"
    write_triangles_json(destination, genes, (32, 32))
    payload = read_json(destination)

    assert payload["active_count"] == 3
    assert payload["active_count"] == active_count(genes)
    assert [entry["active"] for entry in payload["triangles"]] == [True, False, True, False, True, False]


def test_run_json_records_the_effective_configuration_not_the_parser_defaults(
    tmp_path, run_slice0, read_json
) -> None:
    payload = read_json(run_slice0(tmp_path / "run", triangles=8, population=4, canvas=32) / "run.json")
    config = payload["config"]

    assert (config["triangles"], config["population"], config["canvas"]) == (8, 4, 32)
    # Parser defaults are 30 / 8 / 128 — none of them may leak into the archive.
    assert (config["triangles"], config["population"], config["canvas"]) != (30, 8, 128)
    assert config["image"] == "assets/flag_ar.png"
    assert set(payload["versions"]) == {"python", "numpy", "pillow"}
    assert isinstance(payload["seed"], int)


def test_metrics_csv_header_matches_the_stable_schema_and_holds_one_row(tmp_path, run_slice0) -> None:
    rows = _rows(run_slice0(tmp_path / "run"))

    assert rows[0] == METRICS_COLUMNS
    assert len(rows) == 2, "Slice 0 scores one generation and must emit exactly one data row"


@pytest.mark.parametrize("population", [1, 4, 7])
def test_reported_renders_equal_the_chromosomes_actually_rasterized(tmp_path, run_slice0, population) -> None:
    # ROADMAP Success Criterion 4 is about the *reported* count, so this reads
    # the number back out of the file rather than off the in-memory attribute.
    rows = _rows(run_slice0(tmp_path / f"pop{population}", population=population))
    reported = dict(zip(rows[0], rows[1]))

    assert int(reported["renders"]) == population


# --- the run-directory guards ----------------------------------------------


def test_an_existing_populated_directory_is_never_silently_reused(tmp_path, project_root) -> None:
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "best.png").write_bytes(b"prior evidence")

    with pytest.raises(RunDirError) as excinfo:
        prepare_run_dir(destination, project_root, allow_outside=True)
    assert str(destination.resolve()) in str(excinfo.value)

    assert prepare_run_dir(destination, project_root, force=True, allow_outside=True) == destination.resolve()
    assert (destination / "best.png").read_bytes() == b"prior evidence"


def test_an_empty_existing_directory_is_accepted(tmp_path, project_root) -> None:
    destination = tmp_path / "empty"
    destination.mkdir()
    assert prepare_run_dir(destination, project_root, allow_outside=True) == destination.resolve()


def test_a_destination_outside_the_project_root_needs_allow_outside(tmp_path, project_root) -> None:
    outside = tmp_path / "elsewhere"

    with pytest.raises(RunDirError) as excinfo:
        prepare_run_dir(outside, project_root)
    assert str(outside.resolve()) in str(excinfo.value)
    assert not outside.exists(), "a rejected destination must not be created"

    assert prepare_run_dir(outside, project_root, allow_outside=True) == outside.resolve()


def test_the_cli_refuses_to_overwrite_a_run_and_leaves_it_untouched(tmp_path, run_slice0, capsys) -> None:
    run_dir = run_slice0(tmp_path / "run", seed=7)
    before = {name: (run_dir / name).read_bytes() for name in ARTIFACTS}
    capsys.readouterr()

    with pytest.raises(SystemExit):
        run_slice0(run_dir, seed=8)

    assert str(run_dir.resolve()) in capsys.readouterr().err
    assert {name: (run_dir / name).read_bytes() for name in ARTIFACTS} == before


def test_the_cli_overwrites_only_when_force_is_given(tmp_path, run_slice0) -> None:
    run_dir = run_slice0(tmp_path / "run", seed=7)
    before = (run_dir / "best.png").read_bytes()

    run_slice0(run_dir, seed=8, extra=["--force"])
    assert (run_dir / "best.png").read_bytes() != before


# --- argument validation ----------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--triangles", "0"), ("--population", "0"), ("--canvas", "4")],
)
def test_out_of_range_arguments_are_rejected_by_name_and_value(
    tmp_path, target_image, capsys, flag, value
) -> None:
    argv = [
        "--image", str(target_image),
        "--out", str(tmp_path / "run"),
        "--allow-outside",
        "--seed", "7",
        flag, value,
    ]

    with pytest.raises(SystemExit):
        cli.main(argv)

    message = capsys.readouterr().err
    assert flag in message and value in message
    assert not (tmp_path / "run").exists(), "a rejected run must not create its directory"


def test_a_missing_image_is_rejected_by_path(tmp_path, capsys) -> None:
    missing = tmp_path / "not-an-image.png"
    argv = ["--image", str(missing), "--out", str(tmp_path / "run"), "--allow-outside", "--seed", "7"]

    with pytest.raises(SystemExit):
        cli.main(argv)
    assert str(missing) in capsys.readouterr().err
