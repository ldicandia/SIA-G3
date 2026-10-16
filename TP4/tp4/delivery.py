"""Delivery commands: regenerate every config, check the findings notes against the runs.

    python -m tp4.delivery regenerate [--out-root DIR]   # re-run every config of run_order, then the SOM selection check
    python -m tp4.delivery check [--root DIR]            # check docs/hallazgos notes against DIR/runs (default: TP4/runs)

Both read docs/hallazgos/evidencia.json (override with `--manifest PATH`): the run order of the
run configs, the classification of the configs that are not runs and the full cite list of each
note. The manifest holds data only; it never names a module, a function or a command, so the only
processes started are `python -m tp4 run <config> --out-root DIR` for the configs of `run_order`.

Standard library plus existing tp4 modules; no shell, no dynamic evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from . import findings
from .config import load_config
from .errors import Tp4Error
from .experiments.som_sweep_report import check_selection
from .runs import DEFAULT_RUNS_ROOT, PROJECT_ROOT

MANIFEST_PATH = PROJECT_ROOT / "docs" / "hallazgos" / "evidencia.json"
NOTES_DIR = "docs/hallazgos"
CONFIGS_DIR = "configs"

_CONFIG_SUFFIXES = (".json", ".toml")
_TOP_KEYS = frozenset({"run_order", "not_run", "notes"})
_NOTE_KEYS = frozenset({"md", "cites", "requires", "sections"})
_REQUIRES_KEYS = frozenset({"cite", "equals", "hint"})
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class ManifestError(Tp4Error):
    """The delivery manifest is malformed or inconsistent."""


@dataclass(frozen=True)
class Requirement:
    cite: str
    equals: Any
    hint: str


@dataclass(frozen=True)
class Note:
    md: str
    cites: tuple[str, ...]
    requires: tuple[Requirement, ...]
    sections: tuple[Any, ...]


@dataclass(frozen=True)
class Manifest:
    run_order: tuple[str, ...]
    not_run: dict[str, str]
    notes: tuple[Note, ...]
    experiments: dict[str, str]  # config path -> experiment name, for every run_order config

    @property
    def total_cites(self) -> int:
        return sum(len(note.cites) for note in self.notes)


# --------------------------------------------------------------------------- manifest loading


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ManifestError(f"duplicate key {key!r} in the manifest")
        out[key] = value
    return out


def _check_keys(obj: Any, allowed: frozenset[str], where: str, *, exact: bool = True) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ManifestError(f"{where} must be an object")
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise ManifestError(f"unknown key {unknown[0]!r} in {where}")
    if exact:
        missing = sorted(allowed - set(obj))
        if missing:
            raise ManifestError(f"missing key {missing[0]!r} in {where}")
    return obj


def _relative_path(value: Any, where: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where} must be a non-empty string")
    if value.startswith("/") or "\\" in value or _WIN_DRIVE_RE.match(value):
        raise ManifestError(f"{where}: {value!r} must be a relative path with '/' separators")
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ManifestError(f"{where}: {value!r} must not contain '..'")
    return path


def _config_entry(value: Any, project_root: Path, where: str) -> str:
    path = _relative_path(value, where)
    if path.parent != PurePosixPath(CONFIGS_DIR) or path.suffix not in _CONFIG_SUFFIXES:
        raise ManifestError(f"{where}: {value!r} must be a .json or .toml file directly in {CONFIGS_DIR}/")
    if not (project_root / path).is_file():
        raise ManifestError(f"{where}: {value!r} does not exist")
    return str(path)


def _note_md(value: Any, project_root: Path, where: str) -> str:
    path = _relative_path(value, where)
    if path.parent != PurePosixPath(NOTES_DIR) or path.suffix != ".md":
        raise ManifestError(f"{where}: {value!r} must be a .md file directly in {NOTES_DIR}/")
    if not (project_root / path).is_file():
        raise ManifestError(f"{where}: {value!r} does not exist")
    return str(path)


def _cite_spec(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{where} must be a string")
    file, _, key = value.rpartition(":")
    path = _relative_path(file, f"{where} file") if file else None
    if path is None or not key or path.parts[0] != "runs" or len(path.parts) < 3:
        raise ManifestError(f"{where}: {value!r} must look like runs/<experiment>/<file>:<key>")
    return value


def _unique(values: Sequence[str], where: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ManifestError(f"duplicate entry {value!r} in {where}")
        seen.add(value)


def _requirement(raw: Any, where: str) -> Requirement:
    obj = _check_keys(raw, _REQUIRES_KEYS, where)
    cite = _cite_spec(obj["cite"], f"{where}.cite")
    if isinstance(obj["equals"], (dict, list)) or obj["equals"] is None:
        raise ManifestError(f"{where}.equals must be a JSON scalar")
    if not isinstance(obj["hint"], str) or not obj["hint"]:
        raise ManifestError(f"{where}.hint must be a non-empty string")
    return Requirement(cite=cite, equals=obj["equals"], hint=obj["hint"])


def _note(raw: Any, project_root: Path, where: str) -> Note:
    obj = _check_keys(raw, _NOTE_KEYS, where)
    md = _note_md(obj["md"], project_root, f"{where}.md")
    cites = obj["cites"]
    if not isinstance(cites, list):
        raise ManifestError(f"{where}.cites must be a list")
    cites = [_cite_spec(c, f"{where}.cites[{i}]") for i, c in enumerate(cites)]
    _unique(cites, f"{where}.cites")
    requires = obj["requires"]
    if not isinstance(requires, list):
        raise ManifestError(f"{where}.requires must be a list")
    reqs = tuple(_requirement(r, f"{where}.requires[{i}]") for i, r in enumerate(requires))
    sections = obj["sections"]
    if not isinstance(sections, list):
        raise ManifestError(f"{where}.sections must be a list")
    return Note(md=md, cites=tuple(cites), requires=reqs, sections=tuple(sections))


def load_manifest(path: Path | str, project_root: Path = PROJECT_ROOT) -> Manifest:
    """Parse and validate the delivery manifest; raises ManifestError (a Tp4Error) on any problem."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle, object_pairs_hook=_no_duplicate_keys)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    top = _check_keys(raw, _TOP_KEYS, "the manifest")

    run_order_raw = top["run_order"]
    if not isinstance(run_order_raw, list):
        raise ManifestError("run_order must be a list")
    run_order = [_config_entry(c, project_root, f"run_order[{i}]") for i, c in enumerate(run_order_raw)]
    _unique(run_order, "run_order")

    not_run_raw = top["not_run"]
    if not isinstance(not_run_raw, dict):
        raise ManifestError("not_run must be an object")
    not_run: dict[str, str] = {}
    for key, reason in not_run_raw.items():
        entry = _config_entry(key, project_root, f"not_run[{key!r}]")
        if not isinstance(reason, str) or not reason.strip():
            raise ManifestError(f"not_run[{key!r}] needs a written reason")
        not_run[entry] = reason
    both = sorted(set(run_order) & set(not_run))
    if both:
        raise ManifestError(f"{both[0]!r} is in both run_order and not_run")

    notes_raw = top["notes"]
    if not isinstance(notes_raw, list):
        raise ManifestError("notes must be a list")
    notes = tuple(_note(n, project_root, f"notes[{i}]") for i, n in enumerate(notes_raw))
    _unique([n.md for n in notes], "notes")

    experiments: dict[str, str] = {}
    owner: dict[str, str] = {}
    for config in run_order:
        experiment = load_config(project_root / config).get("experiment")
        if not isinstance(experiment, str) or not experiment:
            raise ManifestError(f"{config} has no experiment name")
        if experiment in owner:
            raise ManifestError(f"{config} and {owner[experiment]} share the experiment {experiment!r}")
        owner[experiment] = config
        experiments[config] = experiment
    return Manifest(
        run_order=tuple(run_order), not_run=not_run, notes=notes, experiments=experiments
    )


# --------------------------------------------------------------------------- regenerate


def run_config(config_path: Path | str, out_root: Path | str, project_root: Path = PROJECT_ROOT) -> int:
    """Run one config as its own `python -m tp4 run` process (no shell); returns the exit code."""
    command = [sys.executable, "-m", "tp4", "run", str(config_path), "--out-root", str(out_root)]
    return subprocess.run(command, cwd=project_root, check=False).returncode


def regenerate(manifest: Manifest, out_root: Path | str, project_root: Path = PROJECT_ROOT) -> int:
    """Re-run every run_order config into `out_root`, then check the SOM selection."""
    out_root = Path(out_root)
    total = len(manifest.run_order)
    for index, config in enumerate(manifest.run_order, start=1):
        print(f"[{index}/{total}] {config}", flush=True)
        code = run_config(project_root / config, out_root, project_root)
        if code != 0:
            print(f"FAILED: {config} (exit {code})", flush=True)
            return code
    messages = check_selection(
        project_root / "configs" / "som.json",
        [out_root / "som_design_sweep", out_root / "som_sweep"],
        out_root / "som",
    )
    if messages:
        for message in messages:
            print(message)
        return 1
    print(f"regenerate ok: {total} configs, som selection ok")
    return 0


# --------------------------------------------------------------------------- check


def _experiment_of(artifact: str) -> str:
    return artifact.split("/")[1]


def _display(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _precondition_problems(note: Note, root: Path) -> int:
    problems = 0
    for req in note.requires:
        file, _, key = req.cite.rpartition(":")
        found: Any = findings._MISSING
        try:
            data = json.loads((root / file).read_text(encoding="utf-8"))
            found = findings._lookup(data, key)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            pass
        same = (
            found is not findings._MISSING
            and isinstance(found, bool) == isinstance(req.equals, bool)
            and found == req.equals
        )
        if not same:
            shown = "absent" if found is findings._MISSING else _display(found)
            print(f"PRECONDITION {file}:{key} = {shown}, expected {_display(req.equals)}: {req.hint}")
            problems += 1
    return problems


def check(manifest: Manifest, root: Path | str, project_root: Path = PROJECT_ROOT) -> int:
    """Check every note against `root`/runs; prints one line per problem, returns 0 or 1."""
    root = Path(root)
    problems = 0

    classified = set(manifest.run_order) | set(manifest.not_run)
    configs_dir = project_root / CONFIGS_DIR
    for entry in sorted(configs_dir.iterdir()) if configs_dir.is_dir() else []:
        if entry.is_file() and entry.suffix in _CONFIG_SUFFIXES:
            rel = f"{CONFIGS_DIR}/{entry.name}"
            if rel not in classified:
                print(f"UNCLASSIFIED config: {rel}")
                problems += 1

    experiments = set(manifest.experiments.values())
    for note in manifest.notes:
        text = (project_root / note.md).read_text(encoding="utf-8")
        named: list[tuple[str, str]] = [(a, note.md) for a in findings.referenced_artifacts(text)]
        named += [(c.rpartition(":")[0], f"{note.md} cite") for c in note.cites]
        named += [(r.cite.rpartition(":")[0], f"{note.md} requires") for r in note.requires]
        reported: set[tuple[str, str]] = set()
        for artifact, where in named:
            experiment = _experiment_of(artifact)
            if experiment not in experiments and (experiment, where) not in reported:
                reported.add((experiment, where))
                print(f"NO CONFIG for runs/{experiment} ({where})")
                problems += 1
        problems += _precondition_problems(note, root)

    for note in manifest.notes:
        print(f"== {note.md}")
        argv = [str(project_root / note.md), "--root", str(root)]
        for cite in note.cites:
            argv += ["--cite", cite]
        if findings.main(argv) != 0:
            problems += 1

    if problems:
        print(f"delivery FAILED: {problems} problem(s)")
        return 1
    print(
        f"delivery ok: {len(manifest.notes)} notes, {len(manifest.run_order)} configs, "
        f"{manifest.total_cites} cites"
    )
    return 0


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tp4.delivery", description="Regenerate every config and check the findings notes."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    regen = sub.add_parser("regenerate", help="re-run every config of run_order, then the SOM selection check")
    regen.add_argument("--out-root", type=Path, default=DEFAULT_RUNS_ROOT, help="receives <experiment>/ dirs (default: TP4/runs)")
    chk = sub.add_parser("check", help="check the notes against DIR/runs")
    chk.add_argument("--root", type=Path, default=PROJECT_ROOT, help="runs are read from DIR/runs (default: TP4/)")
    for command in (regen, chk):
        command.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="delivery manifest (default: docs/hallazgos/evidencia.json)")
        command.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="tree holding configs/ and docs/hallazgos/ (default: TP4/; used by tests)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root
    try:
        manifest = load_manifest(args.manifest, project_root)
    except Tp4Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        if args.command == "regenerate":
            return regenerate(manifest, args.out_root, project_root)
        return check(manifest, args.root, project_root)
    except Tp4Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
