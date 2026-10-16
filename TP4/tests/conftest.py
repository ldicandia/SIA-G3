"""Shared fixtures: configs derive from configs/eda.json, never hand-written in full."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tp4.runs import PROJECT_ROOT

REMOVE = "__REMOVE__"


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot encode {value!r} as TOML")


def _toml_dumps(data: dict) -> str:
    lines = [f"{k} = {_toml_value(v)}" for k, v in data.items() if not isinstance(v, dict)]
    for key, table in data.items():
        if isinstance(table, dict):
            lines.append("")
            lines.append(f"[{key}]")
            lines.extend(f"{k} = {_toml_value(v)}" for k, v in table.items())
    return "\n".join(lines) + "\n"


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def out_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def write_config(tmp_path: Path):
    """write_config(base='configs/eda.json', fmt='json', name=None, **overrides) -> Path.

    Overrides may use dotted keys (`**{"params.outlier_iqr_k": 2.0}`); the value
    "__REMOVE__" deletes the key.
    """
    counter = {"n": 0}

    def _write(base: str = "configs/eda.json", fmt: str = "json", **overrides) -> Path:
        config = json.loads((PROJECT_ROOT / base).read_text(encoding="utf-8"))
        for dotted, value in overrides.items():
            node = config
            *parents, leaf = dotted.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            if value == REMOVE:
                node.pop(leaf, None)
            else:
                node[leaf] = value
        counter["n"] += 1
        path = tmp_path / f"cfg{counter['n']}.{fmt}"
        if fmt == "json":
            path.write_text(json.dumps(config), encoding="utf-8")
        else:
            path.write_text(_toml_dumps(config), encoding="utf-8")
        return path

    _write.REMOVE = REMOVE
    return _write


@pytest.fixture
def csv_rows():
    """csv_rows(path) -> number of data rows (stdlib csv, so tests never hardcode 50)."""

    def _count(path) -> int:
        with open(path, newline="", encoding="utf-8") as handle:
            return sum(1 for _ in csv.reader(handle)) - 1

    return _count
