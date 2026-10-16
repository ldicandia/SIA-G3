"""Read-only loading of datos/creadores.csv.

Rows and columns are never dropped or modified (EDA-01 transparency): missing
values are allowed and reported by the EDA; outliers are only flagged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .errors import DataError
from .runs import resolve_project_path

ID_COLUMNS = ("creator_id", "name")


def load_creators(path: str | Path) -> pd.DataFrame:
    """Load and validate the creators CSV. Never writes anything."""
    p = resolve_project_path(path)
    if not p.is_file():
        raise DataError(f"data file not found: {p.name} ({path})")
    try:
        df = pd.read_csv(p, encoding="utf-8")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise DataError(f"cannot read data file {p.name}: {exc}") from exc

    if len(df) == 0:
        raise DataError(f"data file {p.name} has zero data rows")
    missing = [c for c in ID_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"data file {p.name} is missing required column(s): {missing}")
    if df["creator_id"].isna().any() or df["creator_id"].duplicated().any():
        dup = df.loc[df["creator_id"].duplicated(keep=False), "creator_id"].tolist()
        raise DataError(f"creator_id must be present and unique in {p.name}; offending: {dup[:5]}")

    for col in df.columns:
        if col in ID_COLUMNS:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = converted.isna() & df[col].notna()
        if bad.any():
            first = df.loc[bad, "creator_id"].iloc[0]
            raise DataError(
                f"column {col!r} in {p.name} has non-numeric values (first at creator_id {first})"
            )
        df[col] = converted
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Non-ID columns in CSV order."""
    return [c for c in df.columns if c not in ID_COLUMNS]
