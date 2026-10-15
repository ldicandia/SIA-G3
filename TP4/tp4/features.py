"""The declared feature set: which numeric columns every Ej. 1 model trains on.

`configs/feature_set.json` is the single decision (columns dropped, each with a
reason). PCA, the Phase 2 SOM and the Phase 3 Oja index all go through
`select_features`/`load_design`, so they cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import ID_COLUMNS, load_creators, numeric_columns
from .errors import ConfigError
from .runs import resolve_project_path

TOPIC_COLUMNS = ("topic_football_pct", "topic_makeup_pct", "topic_dogs_pct", "topic_art_pct")
FEATURE_SET_KEYS = frozenset({"drop", "reconstruct"})


@dataclass(frozen=True)
class FeatureSet:
    drop: Mapping[str, str]
    reconstruct: Mapping[str, str] = field(default_factory=dict)
    source: str = ""


@dataclass(frozen=True)
class Design:
    """Raw creators frame plus the selected feature matrix (same row order, no rows dropped)."""

    raw: pd.DataFrame
    X: pd.DataFrame
    feature_set: FeatureSet

    @property
    def ids(self) -> pd.DataFrame:
        return self.raw[list(ID_COLUMNS)]


def load_feature_set(path: str | Path) -> FeatureSet:
    """Read and validate the shape of a feature-set file (no data needed)."""
    p = resolve_project_path(path)
    try:
        with p.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"invalid feature set {p.name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"invalid feature set {p.name}: root must be an object")
    unknown = sorted(set(raw) - FEATURE_SET_KEYS)
    if unknown:
        raise ConfigError(
            f"feature set {p.name}: unknown key(s) {unknown}; allowed: {sorted(FEATURE_SET_KEYS)}"
        )
    drop = raw.get("drop")
    if not isinstance(drop, dict) or not drop:
        raise ConfigError(f"feature set {p.name}: 'drop' must be a non-empty object")
    for col, reason in drop.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigError(f"feature set {p.name}: drop[{col!r}] needs a non-empty reason string")
    reconstruct = raw.get("reconstruct", {})
    if not isinstance(reconstruct, dict):
        raise ConfigError(f"feature set {p.name}: 'reconstruct' must be an object")
    for col, formula in reconstruct.items():
        if col not in drop:
            raise ConfigError(f"feature set {p.name}: reconstruct key {col!r} is not in 'drop'")
        if not isinstance(formula, str) or not formula.strip():
            raise ConfigError(f"feature set {p.name}: reconstruct[{col!r}] must be a non-empty string")
    return FeatureSet(drop=dict(drop), reconstruct=dict(reconstruct), source=str(path))


def selected_columns(df: pd.DataFrame, fs: FeatureSet) -> list[str]:
    """Numeric columns of `df` that are not dropped, in CSV order."""
    numeric = numeric_columns(df)
    for col in fs.drop:
        if col in ID_COLUMNS:
            raise ConfigError(f"feature set {fs.source}: cannot drop ID column {col!r}")
        if col not in numeric:
            raise ConfigError(f"feature set {fs.source}: dropped column {col!r} is not a numeric column of the data")
    return [c for c in numeric if c not in fs.drop]


def select_features(df: pd.DataFrame, fs: FeatureSet) -> pd.DataFrame:
    """The selected feature matrix as a new DataFrame (all rows kept)."""
    return df[selected_columns(df, fs)].copy()


def load_design(data_path: str | Path, feature_set_path: str | Path) -> Design:
    df = load_creators(data_path)
    fs = load_feature_set(feature_set_path)
    return Design(raw=df, X=select_features(df, fs), feature_set=fs)
