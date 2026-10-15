from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tp4.findings import cited_value_present, main, referenced_artifacts


def test_referenced_artifacts_from_relative_links():
    text = (
        "![biplot](../../runs/pca/biplot.png) y `runs/eda/columns.csv`, ver runs/pca/summary.json.\n"
        "![otra vez](../../runs/pca/biplot.png)"
    )
    assert referenced_artifacts(text) == [
        "runs/eda/columns.csv",
        "runs/pca/biplot.png",
        "runs/pca/summary.json",
    ]
    assert referenced_artifacts("sin evidencia") == []


def test_float_accepts_decimal_point_or_comma():
    assert cited_value_present("PC1 explica 53.2% de la varianza", 53.2)
    assert cited_value_present("PC1 explica 53,2% de la varianza", 53.2)
    assert not cited_value_present("PC1 explica 53% de la varianza", 53.2)


def test_int_is_not_matched_inside_a_longer_number():
    assert not cited_value_present("hay 16 componentes", 6)
    assert not cited_value_present("el 6.1% restante", 6)
    assert cited_value_present("hay 6 componentes", 6)
    assert cited_value_present("k = 6.", 6)


def test_bool_str_and_list_values():
    assert cited_value_present("estandarizado: True", True)
    assert cited_value_present("sin estandarizar (false)", False)
    assert not cited_value_present("nada", True)
    assert cited_value_present("la variable average_caption_words domina", "average_caption_words")
    assert cited_value_present("a=1 b=2", [1, 2])
    assert not cited_value_present("a=1", [1, 2])
    assert not cited_value_present("x", None)


def _tree(tmp_path, note, summary=None, create=("runs/x/summary.json",)):
    (tmp_path / "note.md").write_text(note, encoding="utf-8")
    for rel in create:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary or {}), encoding="utf-8")
    return tmp_path / "note.md"


def test_missing_artifact_exits_1(tmp_path, capsys):
    note = _tree(tmp_path, "mirá runs/x/summary.json y runs/x/plot.png", create=("runs/x/summary.json",))
    assert main([str(note), "--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "MISSING artifact: runs/x/plot.png" in out
    assert "runs/x/summary.json" not in out


def test_satisfied_tree_exits_0(tmp_path, capsys):
    note = _tree(
        tmp_path,
        "Según runs/x/summary.json, PC1 explica 53,2% con k = 6 y `average_caption_words`.",
        summary={"pct": 53.2, "k": 6, "top": "average_caption_words", "nested": {"v": 6}},
    )
    argv = [str(note), "--root", str(tmp_path)]
    for key in ("pct", "k", "top", "nested.v"):
        argv += ["--cite", f"runs/x/summary.json:{key}"]
    assert main(argv) == 0
    assert capsys.readouterr().out.strip() == "ok: 1 artifacts, 4 cites"


def test_missing_cite_value_and_key_exit_1(tmp_path, capsys):
    note = _tree(tmp_path, "PC1 explica 50%. runs/x/summary.json", summary={"pct": 53.2})
    argv = [str(note), "--root", str(tmp_path), "--cite", "runs/x/summary.json:pct",
            "--cite", "runs/x/summary.json:nope", "--cite", "runs/x/absent.json:pct"]
    assert main(argv) == 1
    out = capsys.readouterr().out
    assert "MISSING cite: runs/x/summary.json:pct=53.2" in out
    assert "MISSING cite: runs/x/summary.json:nope" in out
    assert "MISSING cite: runs/x/absent.json:pct" in out


def test_module_entry_point(tmp_path):
    note = _tree(tmp_path, "ver runs/x/summary.json")
    proc = subprocess.run(
        [sys.executable, "-m", "tp4.findings", str(note), "--root", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("ok:")


@pytest.mark.parametrize("bad", ["sin-dos-puntos", ":k", "f.json:"])
def test_malformed_cite_spec_exits_1(tmp_path, bad):
    note = _tree(tmp_path, "nada")
    assert main([str(note), "--root", str(tmp_path), "--cite", bad]) == 1
