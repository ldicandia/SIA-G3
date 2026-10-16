"""Numpy vs cupy equivalence and timing report for the optional CUDA backend.

Every selected config is run twice through the normal runner (`backend` overridden to numpy and
to cupy, plots off) and the two run directories are compared: `summary.json` and every `*.csv`.
Discrete values (integers, booleans, labels, None) must match exactly; floating-point values must
agree within `rtol`/`atol` (GPU reductions add in a different order, so bit-identity is not
expected). `backend_resolved` is the one summary key allowed to differ.

Two hermetic probes run on synthetic data against the real array module: the lowest-index
tie-breaking of `argmin`, and a short Kohonen training started from duplicated prototypes.

CLI:

    python -m tp4.analysis.backend_compare [--configs PATH ...] [--out-root DIR]
                                           [--json PATH] [--rtol R] [--atol A]

Exit codes: 0 every config and probe equivalent; 1 cupy unavailable (the reason is printed as
`backend unavailable: ...`; nothing ran); 2 usage or config error (`error: ...` on stderr);
3 any mismatch, failed probe or failed cupy run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tp4.backend import resolve_backend, to_numpy
from tp4.config import load_config
from tp4.errors import ConfigError, Tp4Error
from tp4.models.hopfield import HopfieldConfig, recall, train
from tp4.models.oja import OjaConfig, train_oja
from tp4.models.som import SomConfig, train_som
from tp4.runner import execute, validate_config
from tp4.runs import resolve_project_path

DEFAULT_CONFIGS = (
    "configs/som.json",
    "configs/oja.json",
    "configs/hopfield_letters.json",
    "configs/hopfield_qr.json",
)
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9
IGNORED_SUMMARY_KEYS = frozenset({"backend_resolved"})
MAX_LISTED = 10

_INT_CELL = re.compile(r"^[+-]?\d+$")


# --------------------------------------------------------------------------- comparison


def _numbers_close(x: float, y: float, rtol: float, atol: float) -> tuple[bool, float]:
    if math.isnan(x) and math.isnan(y):
        return True, 0.0
    if math.isnan(x) or math.isnan(y):
        return False, math.inf
    if x == y:
        return True, 0.0
    diff = abs(x - y)
    if math.isinf(diff):
        return False, math.inf
    return diff <= atol + rtol * max(abs(x), abs(y)), diff


def _compare_cells(x: str, y: str, rtol: float, atol: float) -> tuple[bool, float | None]:
    """(equal, abs diff when both cells are numbers); integers and text must match exactly."""
    if x == y:
        return True, 0.0
    if _INT_CELL.match(x) and _INT_CELL.match(y):
        return False, float(abs(int(x) - int(y)))
    try:
        fx, fy = float(x), float(y)
    except ValueError:
        return False, None
    return _numbers_close(fx, fy, rtol, atol)


def _compare_values(a: Any, b: Any, where: str, rtol: float, atol: float, out: list[str], diffs: list[float]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{where}.{key}: present only in the {'numpy' if key in a else 'cupy'} run")
            else:
                _compare_values(a[key], b[key], f"{where}.{key}", rtol, atol, out, diffs)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{where}: list length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _compare_values(x, y, f"{where}[{i}]", rtol, atol, out, diffs)
        return
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if not (type(a) is type(b) and a == b):
            out.append(f"{where}: {a!r} vs {b!r}")
        return
    if isinstance(a, int) and isinstance(b, int):
        if a != b:
            out.append(f"{where}: {a!r} vs {b!r}")
            diffs.append(float(abs(a - b)))
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        ok, diff = _numbers_close(float(a), float(b), rtol, atol)
        if math.isfinite(diff):
            diffs.append(diff)
        if not ok:
            out.append(f"{where}: {a!r} vs {b!r} (abs diff {diff:.3e})")
        return
    if a != b:
        out.append(f"{where}: {a!r} vs {b!r}")


def _compare_csv(name: str, path_a: Path, path_b: Path, rtol: float, atol: float, out: list[str], diffs: list[float]) -> None:
    with path_a.open(newline="", encoding="utf-8") as fa, path_b.open(newline="", encoding="utf-8") as fb:
        rows_a, rows_b = list(csv.reader(fa)), list(csv.reader(fb))
    if not rows_a or not rows_b:
        if rows_a != rows_b:
            out.append(f"{name}: one of the files is empty")
        return
    if rows_a[0] != rows_b[0]:
        out.append(f"{name}: columns differ ({rows_a[0]} vs {rows_b[0]})")
        return
    if len(rows_a) != len(rows_b):
        out.append(f"{name}: rows {len(rows_a) - 1} vs {len(rows_b) - 1}")
        return
    header = rows_a[0]
    bad = 0
    for r, (ra, rb) in enumerate(zip(rows_a[1:], rows_b[1:], strict=True), start=1):
        if len(ra) != len(rb):
            out.append(f"{name} row {r}: {len(ra)} vs {len(rb)} cells")
            continue
        for col, (x, y) in enumerate(zip(ra, rb, strict=True)):
            ok, diff = _compare_cells(x, y, rtol, atol)
            if diff is not None and math.isfinite(diff):
                diffs.append(diff)
            if not ok:
                bad += 1
                if bad <= 3:  # the first few offending cells are enough to diagnose
                    out.append(f"{name} row {r} column {header[col]}: {x!r} vs {y!r}")
    if bad > 3:
        out.append(f"{name}: {bad - 3} more differing cells")


def compare_run_dirs(numpy_dir: Path, cupy_dir: Path, rtol: float, atol: float) -> tuple[float, list[str]]:
    """(max abs numeric difference, mismatch messages) between two run directories."""
    numpy_dir, cupy_dir = Path(numpy_dir), Path(cupy_dir)
    mismatches: list[str] = []
    diffs: list[float] = []

    names = ["summary.json"] + sorted(
        {p.name for d in (numpy_dir, cupy_dir) for p in d.glob("*.csv")}
    )
    for name in names:
        pa, pb = numpy_dir / name, cupy_dir / name
        if not pa.is_file() or not pb.is_file():
            side = "cupy" if pa.is_file() else "numpy"
            mismatches.append(f"{name}: missing in the {side} run")
            continue
        if name == "summary.json":
            sa = json.loads(pa.read_text(encoding="utf-8"))
            sb = json.loads(pb.read_text(encoding="utf-8"))
            for key in IGNORED_SUMMARY_KEYS:
                sa.pop(key, None)
                sb.pop(key, None)
            _compare_values(sa, sb, "summary.json", rtol, atol, mismatches, diffs)
        else:
            _compare_csv(name, pa, pb, rtol, atol, mismatches, diffs)
    return (max(diffs) if diffs else 0.0), mismatches


# --------------------------------------------------------------------------- probes


def _probe_argmin_ties(xp, rtol: float, atol: float) -> dict[str, Any]:
    """argmin must return the lowest index among repeated minima (small and large vectors)."""
    cases = {
        "small": np.array([3.0, 1.0, 1.0, 2.0, 1.0]),
        "large": np.tile(np.array([5.0, 2.0, 7.0, 2.0]), 4096),
    }
    got, ok = {}, True
    for label, vec in cases.items():
        expected = int(np.argmin(vec))
        found = int(xp.argmin(xp.asarray(vec)))
        got[label] = found
        ok = ok and found == expected
    return {"name": "argmin_ties", "ok": ok, "detail": f"device argmin {got}, lowest-index rule expects 1 on both"}


def _probe_som_duplicate_prototypes(xp, rtol: float, atol: float) -> dict[str, Any]:
    """Kohonen from duplicated initial prototypes (64 neurons, 30 samples) must agree across backends."""
    X = np.random.default_rng(0).normal(size=(30, 6))
    cfg = SomConfig(rows=8, cols=8, init="sample", epochs=20)
    host = train_som(X, cfg, np.random.default_rng(1))
    dev = train_som(X, cfg, np.random.default_rng(1), xp=xp)
    wa, wb = np.asarray(host.weights), np.asarray(dev.weights)
    diff = float(np.max(np.abs(wa - wb)))
    ok = bool(np.allclose(wa, wb, rtol=rtol, atol=atol))
    return {"name": "som_duplicate_prototypes", "ok": ok, "detail": f"max abs weight diff {diff:.3e}"}


# --------------------------------------------------------------------------- backend info


def _warm_up(xp) -> None:
    """Touch the kernels the runs use so context creation and compilation are not timed."""
    a = xp.asarray(np.arange(16, dtype=np.float64).reshape(4, 4))
    b = a @ a
    xp.where(b > 3.0, b, 0.0)
    xp.exp(-a)
    xp.argmin(a, axis=1)
    a.sum(axis=1)
    to_numpy(xp.linalg.eigvalsh(a + a.T))
    to_numpy(b)
    # One tiny pass through each model so every kernel it uses is compiled before the timers start.
    data = np.random.default_rng(0).normal(size=(6, 4))
    train_som(data, SomConfig(rows=2, cols=2, epochs=1), np.random.default_rng(0), xp=xp)
    train_oja(data, OjaConfig(epochs=1), np.random.default_rng(0), reference=np.ones(4), xp=xp)
    patterns = np.where(np.random.default_rng(1).normal(size=(2, 8)) > 0, 1, -1).astype(np.int8)
    weights = train(patterns, "hebb")
    for update in ("async", "sync"):
        recall(
            weights, patterns, HopfieldConfig(update=update, max_iter=3),
            np.random.default_rng(0), trace=(update == "async"), xp=xp,
        )


def _device_info(xp) -> dict[str, Any]:
    info: dict[str, Any] = {"device": None, "cupy_version": None, "cuda_runtime": None, "cuda_driver": None}
    try:
        name = xp.cuda.runtime.getDeviceProperties(0)["name"]
        info["device"] = name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        pass
    for key, getter in (
        ("cupy_version", lambda: xp.__version__),
        ("cuda_runtime", lambda: xp.cuda.runtime.runtimeGetVersion()),
        ("cuda_driver", lambda: xp.cuda.runtime.driverGetVersion()),
    ):
        try:
            info[key] = getter()
        except Exception:
            pass
    return info


# --------------------------------------------------------------------------- driver


def _load(path: str) -> tuple[Path, dict[str, Any]]:
    p = resolve_project_path(path)
    config = load_config(p)
    if config.get("kind") == "sweep":
        raise ConfigError(f"{path}: sweep configs are not supported (each child keeps its own base backend)")
    config["plots"] = False
    return p, config


def _run_once(path: Path, config: dict[str, Any], backend: str, out_dir: Path) -> tuple[dict[str, Any], float]:
    cfg = validate_config({**config, "backend": backend})
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    t0 = time.perf_counter()
    summary = execute(cfg, out_dir, config_path=path)
    return summary, time.perf_counter() - t0


def _fmt_seconds(v: float | None) -> str:
    return "n/a" if v is None else f"{v:9.2f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tp4.analysis.backend_compare",
        description="Compare numpy and cupy runs of the same configs (equivalence and timings).",
    )
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS), metavar="PATH")
    parser.add_argument("--out-root", type=Path, default=None, help="keep the run dirs here (default: a temp dir)")
    parser.add_argument("--json", type=Path, default=None, metavar="PATH", help="write the report as JSON")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        loaded = [_load(p) for p in args.configs]
    except Tp4Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backend = resolve_backend("cupy")
    if backend.name != "cupy":
        print(f"backend unavailable: {backend.fallback_reason}")
        return 1
    xp = backend.xp
    try:
        _warm_up(xp)
    except Exception as exc:
        print(f"backend unavailable: warm-up failed: {type(exc).__name__}: {exc}")
        return 1

    info = _device_info(xp)
    print(
        f"device: {info['device']}  cupy: {info['cupy_version']}  "
        f"CUDA runtime: {info['cuda_runtime']}  driver: {info['cuda_driver']}"
    )

    tmp = None
    if args.out_root is None:
        tmp = tempfile.TemporaryDirectory(prefix="tp4-backend-compare-")
        root = Path(tmp.name)
    else:
        root = args.out_root
    try:
        entries, failures = _run_configs(loaded, root, args.rtol, args.atol)
    except Tp4Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if tmp is not None:
            tmp.cleanup()

    probes = []
    for probe in (_probe_argmin_ties, _probe_som_duplicate_prototypes):
        try:
            result = probe(xp, args.rtol, args.atol)
        except Exception as exc:
            result = {"name": getattr(probe, "__name__", "probe"), "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        probes.append(result)
        print(f"probe {result['name']}: {'ok' if result['ok'] else 'FAILED'} ({result['detail']})")
        if not result["ok"]:
            failures.append(f"probe {result['name']}: {result['detail']}")

    report = {
        **info,
        "rtol": args.rtol,
        "atol": args.atol,
        "configs": entries,
        "probes": probes,
        "ok": not failures,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if failures:
        print("backend_compare FAILED: " + "; ".join(failures))
        return 3
    print(f"backend_compare ok: {len(entries)} configs equivalent")
    return 0


def _run_configs(
    loaded: list[tuple[Path, dict[str, Any]]], root: Path, rtol: float, atol: float
) -> tuple[list[dict[str, Any]], list[str]]:
    print(f"{'config':<22} {'numpy s':>9} {'cupy s':>9} {'speedup':>8} {'max diff':>10}  verdict")
    entries: list[dict[str, Any]] = []
    failures: list[str] = []
    for path, config in loaded:
        stem = path.stem
        numpy_dir, cupy_dir = root / stem / "numpy", root / stem / "cupy"
        mismatches: list[str] = []
        np_summary, np_seconds = _run_once(path, config, "numpy", numpy_dir)
        cp_seconds: float | None = None
        max_diff = 0.0
        try:
            cp_summary, cp_seconds = _run_once(path, config, "cupy", cupy_dir)
        except Tp4Error:
            raise
        except Exception as exc:  # a real-cupy-only incompatibility: report it, do not hide it
            traceback.print_exc(file=sys.stderr)
            mismatches.append(f"cupy run failed: {type(exc).__name__}: {exc}")
        else:
            for label, summary, want in (("numpy", np_summary, "numpy"), ("cupy", cp_summary, "cupy")):
                got = summary.get("backend_resolved", want)
                if got != want:
                    mismatches.append(f"{label} run resolved backend {got!r}, expected {want!r}")
            max_diff, found = compare_run_dirs(numpy_dir, cupy_dir, rtol, atol)
            mismatches.extend(found)
        equivalent = not mismatches
        speedup = (np_seconds / cp_seconds) if cp_seconds else None
        print(
            f"{stem:<22} {_fmt_seconds(np_seconds)} {_fmt_seconds(cp_seconds)} "
            f"{'n/a' if speedup is None else f'{speedup:7.2f}x':>8} {max_diff:10.2e}  "
            f"{'equivalent' if equivalent else 'MISMATCH'}"
        )
        for line in mismatches[:MAX_LISTED]:
            print(f"    {line}")
        if not equivalent:
            failures.append(f"{stem}: {len(mismatches)} mismatch(es), first: {mismatches[0]}")
        entries.append(
            {
                "config": stem,
                "numpy_seconds": round(np_seconds, 3),
                "cupy_seconds": None if cp_seconds is None else round(cp_seconds, 3),
                "speedup": None if speedup is None else round(speedup, 3),
                "max_abs_diff": max_diff,
                "equivalent": equivalent,
                "n_mismatches": len(mismatches),
                "mismatches": mismatches[:MAX_LISTED],
            }
        )
    return entries, failures


if __name__ == "__main__":
    raise SystemExit(main())
