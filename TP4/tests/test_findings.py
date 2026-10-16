from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tp4.findings import cited_value_present, main, referenced_artifacts
from tp4.findings import section_text


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


# ----- --section: checks scoped to one heading's section

SECTIONED = (
    "# Título\n"
    "intro runs/x/intro.json\n"
    "## A\n"
    "valor 53,2 en A runs/x/summary.json\n"
    "### A.1\n"
    "sub de A con 7\n"
    "```bash\n"
    "# comentario que no es un título\n"
    ".venv/bin/python -m tp4 run configs/a.json\n"
    "```\n"
    "seguimos en A con 9\n"
    "## B\n"
    "solo en B: 11 y runs/x/otro.json\n"
)


def test_section_text_runs_to_the_next_heading_of_same_or_higher_level():
    section = section_text(SECTIONED, "## A")
    assert section.splitlines()[0] == "## A"
    assert "### A.1" in section and "sub de A con 7" in section
    assert "seguimos en A con 9" in section
    assert "## B" not in section and "solo en B" not in section
    assert "intro" not in section


def test_section_text_ignores_hash_lines_inside_fenced_code_blocks():
    section = section_text(SECTIONED, "## A")
    assert "# comentario que no es un título" in section
    assert "seguimos en A con 9" in section


def test_section_text_of_a_subsection_stops_at_the_next_same_level_heading():
    text = "## A\n### A.1\nuno\n### A.2\ndos\n## B\n"
    assert section_text(text, "### A.1").strip().splitlines() == ["### A.1", "uno"]


def test_section_text_handles_crlf_and_trailing_spaces():
    text = "## A  \r\nvalor 1\r\n## B\r\notro\r\n"
    section = section_text(text, "## A")
    assert "valor 1" in section and "otro" not in section


def test_section_text_absent_heading_is_none():
    assert section_text(SECTIONED, "## C") is None
    assert section_text(SECTIONED, "##A") is None


def test_section_scopes_cites_and_artifacts(tmp_path, capsys):
    note = tmp_path / "note.md"
    note.write_text(SECTIONED, encoding="utf-8")
    for rel, data in (
        ("runs/x/summary.json", {"a": 53.2, "b": 11}),
        ("runs/x/otro.json", {}),
        ("runs/x/intro.json", {}),
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    base = [str(note), "--root", str(tmp_path), "--section", "## A"]
    assert main(base + ["--cite", "runs/x/summary.json:a"]) == 0
    assert capsys.readouterr().out.strip() == "ok: 1 artifacts, 1 cites (section ## A)"
    # 11 appears only in section B
    assert main(base + ["--cite", "runs/x/summary.json:b"]) == 1
    assert "MISSING cite: runs/x/summary.json:b=11" in capsys.readouterr().out
    # without --section the same cite is satisfied by the whole note
    assert main([str(note), "--root", str(tmp_path), "--cite", "runs/x/summary.json:b"]) == 0
    capsys.readouterr()
    # an artifact named only outside the section is not checked for that section
    (tmp_path / "runs/x/otro.json").unlink()
    assert main(base) == 0
    assert main([str(note), "--root", str(tmp_path), "--section", "## B"]) == 1
    assert "MISSING artifact: runs/x/otro.json" in capsys.readouterr().out


def test_section_absent_heading_exits_1(tmp_path, capsys):
    note = tmp_path / "note.md"
    note.write_text(SECTIONED, encoding="utf-8")
    assert main([str(note), "--root", str(tmp_path), "--section", "## C"]) == 1
    assert "MISSING section: ## C" in capsys.readouterr().out
