"""The declared feature set: which numeric columns every Ej. 1 model trains on.

`configs/feature_set.json` is the single decision (columns dropped, each with a
reason). PCA, the Phase 2 SOM and the Phase 3 Oja index all go through
`select_features`/`load_design`, so they cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

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
    numeric = [c for c in numeric_columns(df) if is_numeric_dtype(df[c])]
    for col in fs.drop:
        if col in ID_COLUMNS:
            raise ConfigError(f"feature set {fs.source}: cannot drop ID column {col!r}")
        if col not in numeric:
            raise ConfigError(f"feature set {fs.source}: dropped column {col!r} is not a numeric column of the data")
    return [c for c in numeric if c not in fs.drop]


def select_features(df: pd.DataFrame, fs: FeatureSet) -> pd.DataFrame:
    """The selected feature matrix as a new DataFrame (all rows kept).

    This is the enforcement seam for PCA, the Phase 2 SOM and the Phase 3 Oja index:
    a feature set that leaves a holding identity intact is refused with a ConfigError.
    """
    columns = selected_columns(df, fs)
    validate_feature_set(df, fs)
    return df[columns].copy()


def load_design(data_path: str | Path, feature_set_path: str | Path) -> Design:
    df = load_creators(data_path)
    fs = load_feature_set(feature_set_path)
    return Design(raw=df, X=select_features(df, fs), feature_set=fs)


# --------------------------------------------------------------------------- identities


@dataclass(frozen=True)
class Identity:
    """An exact (up to `tolerance`) relation between columns documented in the assignment."""

    name: str
    kind: str  # ratio | sum | product | sum_to_100
    target: str | None
    members: tuple[str, ...]
    formula: str
    tolerance: float


_CONTENT_COLUMNS = (
    "content_tutorial_pct",
    "content_review_pct",
    "content_entertainment_pct",
    "content_showcase_pct",
)

IDENTITIES: tuple[Identity, ...] = (
    Identity(
        "price_ars_ratio", "ratio", "price_ars", ("price_ars", "price_usd"),
        "price_ars = 1500 * price_usd", 0.01,
    ),
    Identity(
        "total_engagement_sum", "sum", "total_engagement_per_1000",
        (
            "total_engagement_per_1000", "likes_per_1000", "comments_per_1000",
            "saves_per_1000", "shares_per_1000",
        ),
        "total_engagement_per_1000 = likes_per_1000 + comments_per_1000 + saves_per_1000 + shares_per_1000",
        0.025,
    ),
    Identity(
        "argentina_reach_product", "product", "estimated_argentina_reach",
        ("estimated_argentina_reach", "average_reach", "argentina_audience_pct"),
        "estimated_argentina_reach = average_reach * argentina_audience_pct / 100", 1.0,
    ),
    Identity(
        "sponsored_organic_complement", "sum_to_100", None,
        ("sponsored_posts_pct", "organic_posts_pct"),
        "sponsored_posts_pct + organic_posts_pct = 100", 0.025,
    ),
    Identity(
        "topic_block", "sum_to_100", None, TOPIC_COLUMNS,
        "topic_football_pct + topic_makeup_pct + topic_dogs_pct + topic_art_pct = 100", 0.025,
    ),
    Identity(
        "content_block", "sum_to_100", None, _CONTENT_COLUMNS,
        "content_tutorial_pct + content_review_pct + content_entertainment_pct + content_showcase_pct = 100",
        0.025,
    ),
)

CHECK_COLUMNS = [
    "name", "kind", "columns", "formula", "max_abs_residual", "tolerance", "holds", "broken_by", "member_std",
]


def identity_residuals(df: pd.DataFrame, identity: Identity) -> np.ndarray:
    """Per-row residual of the identity (0 where it holds exactly)."""
    m = identity.members
    if identity.kind == "ratio":
        res = df[m[0]] - 1500.0 * df[m[1]]
    elif identity.kind == "sum":
        res = df[m[0]] - df[list(m[1:])].sum(axis=1, skipna=False)
    elif identity.kind == "product":
        res = df[m[0]] - df[m[1]] * df[m[2]] / 100.0
    elif identity.kind == "sum_to_100":
        res = df[list(m)].sum(axis=1, skipna=False) - 100.0
    else:  # pragma: no cover - IDENTITIES is a closed literal
        raise ValueError(f"unknown identity kind {identity.kind!r}")
    return res.to_numpy(dtype=float)


def check_identities(df: pd.DataFrame, fs: FeatureSet | None = None) -> pd.DataFrame:
    """One row per identity: does it hold on the data, and which dropped column breaks it."""
    dropped = fs.drop if fs is not None else {}
    rows = []
    for ident in IDENTITIES:
        present = all(c in df.columns for c in ident.members)
        if present:
            residuals = np.abs(identity_residuals(df, ident))
            max_res = float(np.max(residuals)) if residuals.size and not np.isnan(residuals).any() else float("nan")
            holds = bool(max_res <= ident.tolerance)
            formula = ident.formula
            std = ";".join(f"{c}={round(float(df[c].std(ddof=1)), 4)}" for c in ident.members)
        else:
            max_res, holds, std = float("nan"), False, ""
            formula = ident.formula + " (columns missing)"
        rows.append(
            {
                "name": ident.name,
                "kind": ident.kind,
                "columns": ";".join(ident.members),
                "formula": formula,
                "max_abs_residual": max_res,
                "tolerance": ident.tolerance,
                "holds": holds,
                "broken_by": ";".join(c for c in ident.members if c in dropped),
                "member_std": std,
            }
        )
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)


def near_redundant_pairs(corr: pd.DataFrame, threshold: float, selected: Sequence[str]) -> pd.DataFrame:
    """Unordered column pairs with |r| >= threshold, strongest first, ties by (column_a, column_b)."""
    cols = list(corr.columns)
    chosen = set(selected)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = float(corr.loc[a, b])
            if np.isnan(r) or abs(r) < threshold:
                continue
            rows.append(
                {"column_a": a, "column_b": b, "r": r, "abs_r": abs(r), "both_selected": a in chosen and b in chosen}
            )
    out = pd.DataFrame(rows, columns=["column_a", "column_b", "r", "abs_r", "both_selected"])
    out = out.sort_values(["abs_r", "column_a", "column_b"], ascending=[False, True, True], kind="stable")
    return out.reset_index(drop=True)


def validate_feature_set(df: pd.DataFrame, fs: FeatureSet) -> None:
    """Refuse a feature set that leaves a holding identity intact (it would be rank deficient)."""
    report = check_identities(df, fs)
    problems = []
    for _, row in report.iterrows():
        if row["holds"] and not row["broken_by"]:
            problems.append(
                f"identity {row['name']!r} holds on the data ({row['formula']}) but no member "
                f"({row['columns'].replace(';', ', ')}) is dropped; drop one of them"
            )
    if problems:
        raise ConfigError(f"feature set {fs.source}: " + "; ".join(problems))
