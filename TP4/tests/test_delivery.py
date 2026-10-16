from __future__ import annotations

import json

import pytest

from tp4 import delivery
from tp4.delivery import MANIFEST_PATH, ManifestError, load_manifest, main
from tp4.runs import PROJECT_ROOT


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _tree(tmp_path, *, note="El valor es 53,2 y runs/a/summary.json.", manifest=None, extra_configs=()):
    """A tmp project root (configs/, docs/hallazgos/, manifest) that is also its own runs root."""
    _write(tmp_path / "configs/a.json", json.dumps({"experiment": "a"}))
    _write(tmp_path / "configs/b.json", json.dumps({"experiment": "b"}))
    for name in extra_configs:
        _write(tmp_path / f"configs/{name}.json", json.dumps({"experiment": name}))
    _write(tmp_path / "docs/hallazgos/ej.md", note)
    _write(tmp_path / "runs/a/summary.json", json.dumps({"pct": 53.2, "flag": True}))
    data = {
        "run_order": ["configs/a.json", "configs/b.json"],
        "not_run": {},
        "notes": [
            {
                "md": "docs/hallazgos/ej.md",
                "cites": ["runs/a/summary.json:pct"],
                "requires": [],
                "sections": [],
            }
        ],
    }
    if manifest:
        manifest(data)
    return _write(tmp_path / "docs/hallazgos/evidencia.json", json.dumps(data))


def _run(tmp_path, command, manifest_path, *extra):
    return main(
        [command, "--manifest", str(manifest_path), "--project-root", str(tmp_path), *extra]
    )


def _check(tmp_path, manifest_path, root=None):
    return _run(tmp_path, "check", manifest_path, "--root", str(root or tmp_path))


def test_satisfied_tree_exits_0(tmp_path, capsys):
    manifest = _tree(tmp_path)
    assert _check(tmp_path, manifest) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert "== docs/hallazgos/ej.md" in out
    assert out[-1] == "delivery ok: 1 notes, 2 configs, 1 cites"


def test_unlisted_config_is_unclassified(tmp_path, capsys):
    manifest = _tree(tmp_path, extra_configs=("c",))
    assert _check(tmp_path, manifest) == 1
    out = capsys.readouterr().out
    assert "UNCLASSIFIED config: configs/c.json" in out
    assert out.strip().splitlines()[-1] == "delivery FAILED: 1 problem(s)"


def test_not_run_config_is_classified(tmp_path, capsys):
    manifest = _tree(
        tmp_path,
        extra_configs=("c",),
        manifest=lambda d: d["not_run"].update({"configs/c.json": "definición, no es una corrida"}),
    )
    assert _check(tmp_path, manifest) == 0
    assert "UNCLASSIFIED" not in capsys.readouterr().out


def test_artifact_without_config_prints_no_config(tmp_path, capsys):
    manifest = _tree(tmp_path, note="El valor es 53,2 y runs/a/summary.json y runs/ghost/x.csv.")
    assert _check(tmp_path, manifest) == 1
    out = capsys.readouterr().out
    assert "NO CONFIG for runs/ghost (docs/hallazgos/ej.md)" in out
    assert "delivery FAILED:" in out.strip().splitlines()[-1]


def test_cite_value_absent_from_note_exits_1(tmp_path, capsys):
    manifest = _tree(tmp_path, note="El valor es 50 y runs/a/summary.json.")
    assert _check(tmp_path, manifest) == 1
    assert "MISSING cite: runs/a/summary.json:pct=53.2" in capsys.readouterr().out


def test_requires_false_prints_precondition_with_hint(tmp_path, capsys):
    def add_requires(data):
        data["notes"][0]["requires"] = [
            {"cite": "runs/a/summary.json:flag", "equals": False, "hint": "Instalá requirements-qr.txt."}
        ]

    manifest = _tree(tmp_path, manifest=add_requires)
    assert _check(tmp_path, manifest) == 1
    out = capsys.readouterr().out
    assert "PRECONDITION runs/a/summary.json:flag = true, expected false: Instalá requirements-qr.txt." in out


def test_requires_absent_file_prints_precondition(tmp_path, capsys):
    def add_requires(data):
        data["notes"][0]["requires"] = [
            {"cite": "runs/a/other.json:flag", "equals": True, "hint": "regenerá"}
        ]

    manifest = _tree(tmp_path, manifest=add_requires)
    assert _check(tmp_path, manifest) == 1
    assert "PRECONDITION runs/a/other.json:flag = absent, expected true: regenerá" in capsys.readouterr().out


def test_requires_satisfied_passes(tmp_path):
    def add_requires(data):
        data["notes"][0]["requires"] = [
            {"cite": "runs/a/summary.json:flag", "equals": True, "hint": "x"}
        ]

    assert _check(tmp_path, _tree(tmp_path, manifest=add_requires)) == 0


def test_root_without_runs_exits_1_without_raising(tmp_path, capsys):
    manifest = _tree(tmp_path)
    empty = tmp_path / "elsewhere"
    empty.mkdir()
    assert _check(tmp_path, manifest, root=empty) == 1
    out = capsys.readouterr().out
    assert "MISSING artifact: runs/a/summary.json" in out
    assert "MISSING cite" in out


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(extra=1),
        lambda d: d["notes"][0].update(extra=1),
        lambda d: d["notes"][0].update(md="/etc/passwd"),
        lambda d: d["run_order"].__setitem__(0, "/abs/configs/a.json"),
        lambda d: d["run_order"].__setitem__(0, "configs/../configs/a.json"),
        lambda d: d["run_order"].__setitem__(0, "other/a.json"),
        lambda d: d["run_order"].__setitem__(0, "configs/missing.json"),
        lambda d: d["run_order"].__setitem__(1, "configs/a.json"),
        lambda d: d["notes"][0].update(md="docs/hallazgos/../hallazgos/ej.md"),
        lambda d: d["notes"][0].update(md="README.md"),
        lambda d: d["notes"][0].update(cites=["configs/a.json:pct"]),
        lambda d: d["notes"][0].update(cites=["runs/a/summary.json:"]),
        lambda d: d["notes"][0].update(cites=["runs/../x/summary.json:pct"]),
        lambda d: d["notes"][0].update(cites=["runs/a/summary.json:pct", "runs/a/summary.json:pct"]),
        lambda d: d["notes"][0].update(requires=[{"cite": "runs/a/summary.json:flag", "equals": True}]),
        lambda d: d["not_run"].update({"configs/a.json": "razón"}),
        lambda d: d["not_run"].update({"configs/b.json": " "}),
    ],
)
def test_invalid_manifest_exits_2(tmp_path, capsys, mutate):
    manifest = _tree(tmp_path, manifest=mutate)
    assert _check(tmp_path, manifest) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ")
    assert "delivery" not in captured.out


def test_two_configs_with_the_same_experiment_are_rejected(tmp_path, capsys):
    manifest = _tree(tmp_path, manifest=lambda d: None)
    _write(tmp_path / "configs/b.json", json.dumps({"experiment": "a"}))
    assert _check(tmp_path, manifest) == 2
    assert "share the experiment 'a'" in capsys.readouterr().err


def test_duplicate_json_key_is_rejected(tmp_path, capsys):
    manifest = _tree(tmp_path)
    manifest.write_text(manifest.read_text(encoding="utf-8").replace('"not_run": {}', '"not_run": {}, "not_run": {}'), encoding="utf-8")
    assert _check(tmp_path, manifest) == 2
    assert "duplicate key" in capsys.readouterr().err


def test_unreadable_manifest_exits_2(tmp_path, capsys):
    assert _check(tmp_path, tmp_path / "nope.json") == 2
    assert capsys.readouterr().err.startswith("error: ")


# ----- regenerate with run_config patched (no real runs)


def test_regenerate_runs_every_config_in_order(tmp_path, monkeypatch, capsys):
    manifest = load_manifest(_tree(tmp_path), tmp_path)
    calls = []
    monkeypatch.setattr(delivery, "run_config", lambda cfg, out, root: calls.append((cfg.name, out)) or 0)
    monkeypatch.setattr(delivery, "check_selection", lambda *a, **k: [])
    assert delivery.regenerate(manifest, tmp_path / "out", tmp_path) == 0
    assert [name for name, _ in calls] == ["a.json", "b.json"]
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "[1/2] configs/a.json" and out[1] == "[2/2] configs/b.json"
    assert out[-1] == "regenerate ok: 2 configs, som selection ok"


def test_run_config_uses_an_argument_list_without_a_shell(tmp_path, monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"], seen["kwargs"] = command, kwargs
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(delivery.subprocess, "run", fake_run)
    assert delivery.run_config(tmp_path / "configs/a.json", tmp_path / "out", tmp_path) == 0
    assert isinstance(seen["command"], list)
    assert seen["command"][1:4] == ["-m", "tp4", "run"]
    assert seen["command"][-2] == "--out-root"
    assert not seen["kwargs"].get("shell")
    assert seen["kwargs"]["cwd"] == tmp_path


# ----- real manifest, tracked files only (no runs needed)


def test_every_real_config_is_classified():
    manifest = load_manifest(MANIFEST_PATH)
    classified = set(manifest.run_order) | set(manifest.not_run)
    on_disk = {f"configs/{p.name}" for p in (PROJECT_ROOT / "configs").iterdir() if p.suffix in (".json", ".toml")}
    assert on_disk == classified
    assert len(manifest.run_order) == 12
    assert set(manifest.not_run) == {"configs/feature_set.json", "configs/som_design_base.json"}


def test_real_manifest_keeps_every_cite_and_the_run_order():
    manifest = load_manifest(MANIFEST_PATH)
    by_md = {note.md: note for note in manifest.notes}
    assert len(by_md["docs/hallazgos/ej1.md"].cites) >= 53
    assert len(by_md["docs/hallazgos/ej2.md"].cites) >= 46
    assert [c.rsplit("/", 1)[1] for c in manifest.run_order] == [
        "eda.json", "pca.json", "pca_scaling_sweep.json", "som_design_sweep.json", "som_sweep.json",
        "som.json", "som_pca_control.json", "oja_sweep.json", "oja.json", "campaigns.json",
        "hopfield_letters.json", "hopfield_qr.json",
    ]
    (requirement,) = by_md["docs/hallazgos/ej2.md"].requires
    assert requirement.cite == "runs/hopfield_qr/summary.json:decoder_available"
    assert requirement.equals is True and "requirements-qr.txt" in requirement.hint


def test_invalid_manifest_class():
    assert issubclass(ManifestError, delivery.Tp4Error)


# ----- Task 2: sections, broken section references, verify, stop-on-failure

SECTION_NOTE = (
    "# Nota\n"
    "## A\n"
    "El valor es 53,2 en A, runs/a/summary.json.\n"
    "```bash\n"
    "# no es un título\n"
    "```\n"
    "## B\n"
    "El indicador es true en B.\n"
)


def _with_sections(data):
    data["notes"][0]["sections"] = [
        {"heading": "## A", "cites": ["runs/a/summary.json:pct"]},
        {"heading": "## B", "cites": ["runs/a/summary.json:flag"]},
    ]


def test_sections_are_checked_and_counted(tmp_path, capsys):
    manifest = _tree(tmp_path, note=SECTION_NOTE, manifest=_with_sections)
    assert _check(tmp_path, manifest) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert "== docs/hallazgos/ej.md :: ## A" in out
    assert "== docs/hallazgos/ej.md :: ## B" in out
    assert out[-1] == "delivery ok: 1 notes, 2 configs, 3 cites"


def test_section_cite_outside_its_section_fails(tmp_path, capsys):
    def misplaced(data):
        data["notes"][0]["sections"] = [{"heading": "## B", "cites": ["runs/a/summary.json:pct"]}]

    manifest = _tree(tmp_path, note=SECTION_NOTE, manifest=misplaced)
    assert _check(tmp_path, manifest) == 1
    assert "MISSING cite: runs/a/summary.json:pct=53.2" in capsys.readouterr().out


def test_absent_section_heading_fails(tmp_path, capsys):
    def absent(data):
        data["notes"][0]["sections"] = [{"heading": "## Z", "cites": []}]

    manifest = _tree(tmp_path, note=SECTION_NOTE, manifest=absent)
    assert _check(tmp_path, manifest) == 1
    assert "MISSING section: ## Z" in capsys.readouterr().out


@pytest.mark.parametrize(
    "section",
    [
        {"heading": "A", "cites": []},
        {"heading": "##A", "cites": []},
        {"heading": "####### A", "cites": []},
        {"heading": "## A"},
        {"heading": "## A", "cites": ["configs/a.json:x"]},
        {"heading": "## A", "cites": [], "extra": 1},
        "## A",
    ],
)
def test_invalid_section_entry_exits_2(tmp_path, capsys, section):
    manifest = _tree(tmp_path, note=SECTION_NOTE, manifest=lambda d: d["notes"][0].update(sections=[section]))
    assert _check(tmp_path, manifest) == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_broken_section_reference_is_reported(tmp_path, capsys):
    note = SECTION_NOTE + "Ver `### Comparación con PC1` para el detalle.\n"
    manifest = _tree(tmp_path, note=note)
    assert _check(tmp_path, manifest) == 1
    out = capsys.readouterr().out
    assert "BROKEN section reference: ### Comparación con PC1 (docs/hallazgos/ej.md)" in out


def test_matching_section_reference_passes(tmp_path, capsys):
    note = SECTION_NOTE + "Ver `## A` y `## B`; un `#hashtag` no es una referencia.\n"
    manifest = _tree(tmp_path, note=note)
    assert _check(tmp_path, manifest) == 0, capsys.readouterr().out


def test_regenerate_stops_at_the_first_failing_config(tmp_path, monkeypatch, capsys):
    manifest = load_manifest(_tree(tmp_path), tmp_path)
    calls = []
    monkeypatch.setattr(delivery, "run_config", lambda cfg, out, root: calls.append(cfg.name) or 2)
    selection = []
    monkeypatch.setattr(delivery, "check_selection", lambda *a, **k: selection.append(1) or [])
    assert delivery.regenerate(manifest, tmp_path / "out", tmp_path) == 2
    assert calls == ["a.json"]
    assert selection == []
    assert "FAILED: configs/a.json (exit 2)" in capsys.readouterr().out


def test_regenerate_reports_selection_mismatch(tmp_path, monkeypatch, capsys):
    manifest = load_manifest(_tree(tmp_path), tmp_path)
    monkeypatch.setattr(delivery, "run_config", lambda cfg, out, root: 0)
    monkeypatch.setattr(delivery, "check_selection", lambda *a, **k: ["MISMATCH som_sweep: k config=1 selected=2"])
    assert delivery.regenerate(manifest, tmp_path / "out", tmp_path) == 1
    assert "MISMATCH som_sweep" in capsys.readouterr().out


def _patch_verify(monkeypatch, regenerate_code, check_code):
    seen = {}

    def fake_regenerate(manifest, out_root, project_root=PROJECT_ROOT):
        seen["out_root"] = out_root
        out_root.mkdir(parents=True, exist_ok=True)
        return regenerate_code

    def fake_check(manifest, root, project_root=PROJECT_ROOT):
        seen["root"] = root
        seen["check_called"] = True
        return check_code

    monkeypatch.setattr(delivery, "regenerate", fake_regenerate)
    monkeypatch.setattr(delivery, "check", fake_check)
    return seen


def test_verify_regenerates_into_a_temp_dir_and_removes_it(tmp_path, monkeypatch, capsys):
    manifest = load_manifest(_tree(tmp_path), tmp_path)
    seen = _patch_verify(monkeypatch, 0, 0)
    assert delivery.verify(manifest, tmp_path) == 0
    assert seen["out_root"] == seen["root"] / "runs"
    assert seen["root"] != tmp_path and not seen["root"].exists()
    out = capsys.readouterr().out
    assert f"verify: clean regeneration in {seen['root']}" in out
    assert out.strip().splitlines()[-1] == "verify ok: clean regeneration matches the notes"


def test_verify_removes_the_temp_dir_when_check_fails(tmp_path, monkeypatch, capsys):
    manifest = load_manifest(_tree(tmp_path), tmp_path)
    seen = _patch_verify(monkeypatch, 0, 1)
    assert delivery.verify(manifest, tmp_path) == 1
    assert not seen["root"].exists()
    assert "verify ok" not in capsys.readouterr().out


def test_verify_skips_check_when_regenerate_fails(tmp_path, monkeypatch):
    manifest = load_manifest(_tree(tmp_path), tmp_path)
    seen = _patch_verify(monkeypatch, 2, 0)
    assert delivery.verify(manifest, tmp_path) == 2
    assert "check_called" not in seen
    assert not seen["out_root"].exists()


def test_verify_subcommand_exists(tmp_path, monkeypatch):
    manifest = _tree(tmp_path)
    seen = _patch_verify(monkeypatch, 0, 0)
    assert main(["verify", "--manifest", str(manifest), "--project-root", str(tmp_path)]) == 0
    assert seen["check_called"]


# ----- real tracked files only (no runs needed)


def test_every_real_run_config_validates_with_distinct_experiments():
    from tp4.config import load_config
    from tp4.runner import validate_config

    manifest = load_manifest(MANIFEST_PATH)
    experiments = []
    for config in manifest.run_order:
        validated = validate_config(load_config(PROJECT_ROOT / config))
        experiments.append(validated["experiment"])
    assert len(set(experiments)) == len(experiments) == 12


def test_every_runs_reference_in_the_real_notes_has_a_config():
    from tp4.findings import referenced_artifacts

    manifest = load_manifest(MANIFEST_PATH)
    experiments = set(manifest.experiments.values())
    for note in manifest.notes:
        text = (PROJECT_ROOT / note.md).read_text(encoding="utf-8")
        named = referenced_artifacts(text) + [c.rpartition(":")[0] for c in note.cites]
        named += [r.cite.rpartition(":")[0] for r in note.requires]
        for artifact in named:
            assert artifact.split("/")[1] in experiments, f"{artifact} in {note.md}"


def test_real_notes_have_no_broken_section_reference():
    manifest = load_manifest(MANIFEST_PATH)
    for note in manifest.notes:
        text = (PROJECT_ROOT / note.md).read_text(encoding="utf-8")
        assert delivery.broken_section_references(text) == [], note.md
