"""Pure EDA functions. They never drop or modify rows (EDA-01 prohibition)."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from ..data import ID_COLUMNS


def describe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One row per CSV column, in CSV order, with basic descriptive statistics."""
    rows = []
    for col in df.columns:
        s = df[col]
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "n": int(len(s)),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "max": None,
        }
        if col not in ID_COLUMNS and is_numeric_dtype(s):
            row.update(
                mean=s.mean(),
                std=s.std(ddof=1),
                min=s.min(),
                median=s.median(),
                max=s.max(),
            )
        rows.append(row)
    return pd.DataFrame(rows)
