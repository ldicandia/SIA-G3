"""Delivery commands: regenerate every config, check the findings notes against the runs.

    python -m tp4.delivery regenerate [--out-root DIR]   # re-run every config of run_order, then the SOM selection check
    python -m tp4.delivery check [--root DIR]            # check docs/hallazgos notes against DIR/runs (default: TP4/runs)
    python -m tp4.delivery figures [--root DIR]          # copy the figures the notes embed from DIR/runs into docs/hallazgos/figuras/
    python -m tp4.delivery verify                        # regenerate into a temporary directory, check against it, remove it

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
import tempfile
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
_SECTION_KEYS = frozenset({"heading", "cites"})
_SECTION_HEADING_RE = re.compile(r"^#{1,6} \S")
_HEADING_LINE_RE = re.compile(r"^#{1,6} ")
_BACKTICK_SPAN_RE = re.compile(r"`([^`\n]+)`")
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:")

FIGURES_DIR = "docs/hallazgos/figuras"
_RUN_LINK_PREFIX = "../../runs/"
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")
_LINK_INNER_RE = re.compile(r'(\S+)(?:\s+"([^"]*)")?')
_COPY_TARGET_RE = re.compile(r"^figuras/([A-Za-z0-9_.-]+\.png)$")
_FIGURE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_FIGURE_MENTION_RE = re.compile(r"figuras/([^\s)\]\"'<>]+\.png)")


class ManifestError(Tp4Error):
    """The delivery manifest is malformed or inconsistent."""


@dataclass(frozen=True)
class Requirement:
    cite: str
    equals: Any
    hint: str


@dataclass(frozen=True)
class Section:
    heading: str
    cites: tuple[str, ...]


@dataclass(frozen=True)
class Note:
    md: str
    cites: tuple[str, ...]
    requires: tuple[Requirement, ...]
    sections: tuple[Section, ...]


@dataclass(frozen=True)
class Manifest:
    run_order: tuple[str, ...]
    not_run: dict[str, str]
    notes: tuple[Note, ...]
    experiments: dict[str, str]  # config path -> experiment name, for every run_order config

    @property
    def total_cites(self) -> int:
        return sum(
            len(note.cites) + sum(len(section.cites) for section in note.sections)
            for note in self.notes
        )


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


def _section(raw: Any, where: str) -> Section:
    obj = _check_keys(raw, _SECTION_KEYS, where)
    heading = obj["heading"]
    if not isinstance(heading, str) or not _SECTION_HEADING_RE.match(heading):
        raise ManifestError(f"{where}.heading must start with 1 to 6 '#' marks and a space")
    cites = obj["cites"]
    if not isinstance(cites, list):
        raise ManifestError(f"{where}.cites must be a list")
    cites = [_cite_spec(c, f"{where}.cites[{i}]") for i, c in enumerate(cites)]
    _unique(cites, f"{where}.cites")
    return Section(heading=heading, cites=tuple(cites))


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
    parsed = tuple(_section(s, f"{where}.sections[{i}]") for i, s in enumerate(sections))
    _unique([s.heading for s in parsed], f"{where}.sections")
    return Note(md=md, cites=tuple(cites), requires=reqs, sections=parsed)


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


def _outside_fences(text: str) -> list[str]:
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            lines.append(line)
    return lines


def broken_section_references(text: str) -> list[str]:
    """Backticked heading references (`### Title`) of `text` that match no heading line of it."""
    lines = _outside_fences(text)
    headings = {line.rstrip() for line in lines if _HEADING_LINE_RE.match(line)}
    broken: list[str] = []
    for line in lines:
        for span in _BACKTICK_SPAN_RE.findall(line):
            if _HEADING_LINE_RE.match(span) and span.rstrip() not in headings and span not in broken:
                broken.append(span)
    return broken


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


# --------------------------------------------------------------------------- figures


def _source_parts(title: str | None) -> list[str] | None:
    """Segments of a `runs/<path>.png` source, or None when it is not a plain relative PNG path."""
    if not title or not title.startswith("runs/") or not title.endswith(".png"):
        return None
    parts = title.split("/")
    if len(parts) < 2 or parts[-1] == ".png":
        return None
    for part in parts:
        if part in (".", "..") or not _FIGURE_SEGMENT_RE.match(part):
            return None
    return parts


def flatten(source: str) -> str:
    """Copy name of a run artifact: drop the leading `runs/`, join the remaining segments with `__`."""
    return "__".join(source.split("/")[1:])


def _scan_image_links(text: str) -> list[tuple[int, re.Match[str]]]:
    """(line index, match) of every image link outside fenced blocks."""
    found: list[tuple[int, re.Match[str]]] = []
    in_fence = False
    for index, line in enumerate(text.splitlines(keepends=True)):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            found.extend((index, match) for match in _IMAGE_RE.finditer(line))
    return found


def _link_parts(match: re.Match[str]) -> tuple[str, str | None] | None:
    """(target, title) of an image link, or None when its parentheses do not parse."""
    inner = _LINK_INNER_RE.fullmatch(match.group(2).strip())
    if inner is None:
        return None
    return inner.group(1), inner.group(2)


def _declared_copy(match: re.Match[str]) -> tuple[str, str] | None:
    """(copy name, source title) when the link is in the copy form, else None."""
    parts = _link_parts(match)
    if parts is None:
        return None
    target, title = parts
    target_match = _COPY_TARGET_RE.match(target)
    if target_match is None or _source_parts(title) is None:
        return None
    name = target_match.group(1)
    if name != flatten(title or ""):
        return None
    return name, title or ""


def _rewrite_run_links(text: str) -> tuple[str, int]:
    """Turn `![a](../../runs/x/y.png)` links into the copy form; returns (new text, links rewritten)."""
    count = 0
    in_fence = False
    out: list[str] = []

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        parts = _link_parts(match)
        if parts is None or not parts[0].startswith(_RUN_LINK_PREFIX) or not parts[0].endswith(".png"):
            return match.group(0)
        source = parts[0][len("../../") :]
        if _source_parts(source) is None:
            return match.group(0)
        count += 1
        return f'![{match.group(1)}](figuras/{flatten(source)} "{source}")'

    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            out.append(_IMAGE_RE.sub(replace, line))
    return "".join(out), count


def _png_names(figures_dir: Path) -> list[str]:
    if not figures_dir.is_dir():
        return []
    return sorted(p.name for p in figures_dir.iterdir() if p.is_file() and p.name.endswith(".png"))


@dataclass
class _Declared:
    copies: dict[str, str]  # copy name -> source title
    bad: list[tuple[str, str]]  # (link text, note md)
    mentions: list[tuple[str, str]]  # (copy name mentioned, note md)


def _collect_figures(manifest: Manifest, texts: dict[str, str]) -> _Declared:
    declared = _Declared(copies={}, bad=[], mentions=[])
    for note in manifest.notes:
        text = texts[note.md]
        for _, match in _scan_image_links(text):
            copy = _declared_copy(match)
            if copy is None or declared.copies.get(copy[0], copy[1]) != copy[1]:
                declared.bad.append((match.group(0), note.md))
                continue
            declared.copies[copy[0]] = copy[1]
        for line in _outside_fences(text):
            for name in _FIGURE_MENTION_RE.findall(line):
                declared.mentions.append((name, note.md))
    return declared


def sync_figures(manifest: Manifest, root: Path | str, project_root: Path = PROJECT_ROOT) -> int:
    """Rewrite run links to the copy form, copy each declared source into docs/hallazgos/figuras/.

    Sources are read only under `root`; writes happen only directly inside FIGURES_DIR (and in
    the notes' image links); only *.png files of FIGURES_DIR that no note declares are deleted.
    """
    root = Path(root)
    rewritten = 0
    texts: dict[str, str] = {}
    for note in manifest.notes:
        note_path = project_root / note.md
        with note_path.open(encoding="utf-8", newline="") as handle:
            text = handle.read()
        new_text, count = _rewrite_run_links(text)
        if new_text != text:
            with note_path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(new_text)
        rewritten += count
        texts[note.md] = new_text

    declared = _collect_figures(manifest, texts)
    problems = 0
    for link, md in declared.bad:
        print(f"BAD figure link: {link} ({md})")
        problems += 1

    figures_dir = project_root / FIGURES_DIR
    copied = 0
    for name, title in sorted(declared.copies.items()):
        source = root / title
        if not source.is_file():
            print(f"MISSING figure source: {title}")
            problems += 1
            continue
        target = figures_dir / name
        data = source.read_bytes()
        if target.is_file() and target.read_bytes() == data:
            continue
        figures_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        copied += 1

    removed = 0
    if not declared.bad:  # an unparsable link makes the declared set unreliable: delete nothing
        for name in _png_names(figures_dir):
            if name not in declared.copies:
                (figures_dir / name).unlink()
                removed += 1
    print(f"figures: {len(declared.copies)} declared, {copied} copied, {rewritten} links rewritten, {removed} removed")
    return 1 if problems else 0


def _figure_problems(manifest: Manifest, root: Path, project_root: Path, texts: dict[str, str]) -> tuple[int, int]:
    """Print one line per figure problem; returns (problems, declared copies)."""
    declared = _collect_figures(manifest, texts)
    problems = 0
    for link, md in declared.bad:
        print(f"BAD figure link: {link} ({md})")
        problems += 1
    figures_dir = project_root / FIGURES_DIR
    for name, title in sorted(declared.copies.items()):
        copy = figures_dir / name
        source = root / title
        if not source.is_file():
            print(f"MISSING figure source: {title}")
            problems += 1
        if not copy.is_file():
            print(f"MISSING figure copy: {name}")
            problems += 1
        elif source.is_file() and copy.read_bytes() != source.read_bytes():
            print(f"STALE figure: {name} differs from {title}")
            problems += 1
    for name in _png_names(figures_dir):
        if name not in declared.copies:
            print(f"ORPHAN figure copy: {name}")
            problems += 1
    for name, md in declared.mentions:
        if name not in declared.copies:
            print(f"UNKNOWN figure reference: {name} ({md})")
            problems += 1
    return problems, len(declared.copies)


# --------------------------------------------------------------------------- check


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
    texts: dict[str, str] = {}
    for note in manifest.notes:
        text = (project_root / note.md).read_text(encoding="utf-8")
        texts[note.md] = text
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
        for span in broken_section_references(text):
            print(f"BROKEN section reference: {span} ({note.md})")
            problems += 1
        problems += _precondition_problems(note, root)

    figure_problems, n_figures = _figure_problems(manifest, root, project_root, texts)
    problems += figure_problems

    for note in manifest.notes:
        print(f"== {note.md}")
        argv = [str(project_root / note.md), "--root", str(root)]
        for cite in note.cites:
            argv += ["--cite", cite]
        if findings.main(argv) != 0:
            problems += 1
        for section in note.sections:
            print(f"== {note.md} :: {section.heading}")
            section_argv = [*argv[:3], "--section", section.heading]
            for cite in section.cites:
                section_argv += ["--cite", cite]
            if findings.main(section_argv) != 0:
                problems += 1

    if problems:
        print(f"delivery FAILED: {problems} problem(s)")
        return 1
    print(
        f"delivery ok: {len(manifest.notes)} notes, {len(manifest.run_order)} configs, "
        f"{manifest.total_cites} cites, {n_figures} figures"
    )
    return 0


# --------------------------------------------------------------------------- verify


def verify(manifest: Manifest, project_root: Path = PROJECT_ROOT) -> int:
    """Regenerate everything into a temporary directory and check the notes against it.

    runs/ is never written; the temporary directory is removed in every outcome.
    """
    with tempfile.TemporaryDirectory(prefix="tp4-verify-") as tmp:
        root = Path(tmp)
        print(f"verify: clean regeneration in {root}", flush=True)
        code = regenerate(manifest, root / "runs", project_root)
        if code != 0:
            return code
        code = check(manifest, root, project_root)
        if code == 0:
            print("verify ok: clean regeneration matches the notes")
        return code


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
    fig = sub.add_parser("figures", help="rewrite run links of the notes and copy each declared figure into docs/hallazgos/figuras/")
    fig.add_argument("--root", type=Path, default=PROJECT_ROOT, help="figures are read from DIR/runs (default: TP4/)")
    ver = sub.add_parser("verify", help="clean regeneration into a temporary directory, then check against it")
    for command in (regen, chk, fig, ver):
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
        if args.command == "verify":
            return verify(manifest, project_root)
        if args.command == "figures":
            return sync_figures(manifest, args.root, project_root)
        return check(manifest, args.root, project_root)
    except Tp4Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
