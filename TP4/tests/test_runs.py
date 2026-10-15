from __future__ import annotations

import json

import pytest

from tp4.errors import RunDirError
from tp4.runs import RUN_MARKER, staged_run_dir, write_json


def _make_run(path, **files):
    path.mkdir(parents=True)
    (path / RUN_MARKER).write_text("{}", encoding="utf-8")
    for name, content in files.items():
        (path / name).write_text(content, encoding="utf-8")


def test_replace_wholesale_removes_stale_files(out_root):
    _make_run(out_root / "exp", stale="old")
    with staged_run_dir(out_root, "exp") as staging:
        (staging / RUN_MARKER).write_text("{}", encoding="utf-8")
        (staging / "fresh.txt").write_text("new", encoding="utf-8")
    assert (out_root / "exp" / "fresh.txt").read_text() == "new"
    assert not (out_root / "exp" / "stale").exists()
    assert not list(out_root.glob(".staging-*"))


def test_refuses_dir_without_marker_and_leaves_it_untouched(out_root):
    foreign = out_root / "exp"
    foreign.mkdir(parents=True)
    (foreign / "precious.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RunDirError):
        with staged_run_dir(out_root, "exp"):
            pytest.fail("body must not run")
    assert (foreign / "precious.txt").read_text() == "keep"
    assert [p.name for p in foreign.iterdir()] == ["precious.txt"]
    assert not list(out_root.glob(".staging-*"))


def test_exception_inside_context_keeps_previous_run_and_removes_staging(out_root):
    _make_run(out_root / "exp", old="previous")
    with pytest.raises(RuntimeError):
        with staged_run_dir(out_root, "exp") as staging:
            (staging / "partial.txt").write_text("x", encoding="utf-8")
            raise RuntimeError("boom")
    assert (out_root / "exp" / "old").read_text() == "previous"
    assert not (out_root / "exp" / "partial.txt").exists()
    assert not list(out_root.glob(".staging-*"))


@pytest.mark.parametrize("name", ["../x", "a/b", "", "Eda", ".hidden", "a" * 65, "x\n"])
def test_rejects_bad_experiment_names(out_root, name):
    with pytest.raises(RunDirError):
        with staged_run_dir(out_root, name):
            pytest.fail("body must not run")
    assert not out_root.exists() or not list(out_root.iterdir())


def test_accepts_64_char_name(out_root):
    with staged_run_dir(out_root, "a" * 64) as staging:
        (staging / RUN_MARKER).write_text("{}", encoding="utf-8")
    assert (out_root / ("a" * 64)).is_dir()


def test_write_json_sorted_keys_trailing_newline_and_nan_to_null(tmp_path):
    path = tmp_path / "x.json"
    write_json(path, {"b": 1, "a": float("nan"), "c": [2.5]})
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert list(json.loads(text)) == ["a", "b", "c"]
    assert json.loads(text)["a"] is None
