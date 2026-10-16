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
