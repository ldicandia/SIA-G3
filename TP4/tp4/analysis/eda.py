"""Pure EDA functions. They never drop or modify rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ..data import ID_COLUMNS

OUTLIER_COLUMNS = ["column", "creator_id", "name", "value", "lower_fence", "upper_fence", "side"]


def tukey_fences(s: pd.Series, k: float) -> tuple[float, float]:
    """(Q1 - k*IQR, Q3 + k*IQR) over the non-null values (linear-interpolation quantiles)."""
    values = s.dropna()
    if values.empty:
        return (float("nan"), float("nan"))
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)


def _is_stats_column(df: pd.DataFrame, col: str) -> bool:
    return col not in ID_COLUMNS and is_numeric_dtype(df[col])


def describe_columns(df: pd.DataFrame, outlier_k: float = 1.5) -> pd.DataFrame:
    """One row per CSV column, in CSV order. Numeric statistics are empty for ID columns."""
    rows = []
    for col in df.columns:
        s = df[col]
        row: dict[str, object] = {
            "column": col,
            "dtype": str(s.dtype),
            "n": int(len(s)),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        stats = dict.fromkeys(
            ["mean", "std", "min", "q1", "median", "q3", "max", "iqr",
             "lower_fence", "upper_fence", "n_outliers"],
            np.nan,
        )
        if _is_stats_column(df, col):
            values = s.dropna()
            lo, hi = tukey_fences(values, outlier_k)
            stats.update(
                mean=values.mean(),
                std=values.std(ddof=1),
                min=values.min(),
                q1=values.quantile(0.25),
                median=values.median(),
                q3=values.quantile(0.75),
                max=values.max(),
                iqr=values.quantile(0.75) - values.quantile(0.25),
                lower_fence=lo,
                upper_fence=hi,
                n_outliers=int(((values < lo) | (values > hi)).sum()),
            )
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows)


def iqr_outliers(df: pd.DataFrame, columns: list[str], k: float) -> pd.DataFrame:
    """Rows outside the Tukey fences (strictly), ordered by column position then creator_id.

    A value exactly equal to a fence is NOT an outlier. Missing values never are.
    The input frame is never modified.
    """
    pieces = []
    for col in columns:
        s = df[col]
        lo, hi = tukey_fences(s, k)
        mask = (s < lo) | (s > hi)
        if not mask.any():
            continue
        sub = df.loc[mask, ["creator_id", "name"]].copy()
        sub.insert(0, "column", col)
        sub["value"] = s[mask].to_numpy()
        sub["lower_fence"] = lo
        sub["upper_fence"] = hi
        sub["side"] = np.where(s[mask].to_numpy() < lo, "low", "high")
        pieces.append(sub.sort_values("creator_id", kind="stable"))
    if not pieces:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in OUTLIER_COLUMNS})
    return pd.concat(pieces, ignore_index=True)[OUTLIER_COLUMNS]


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pearson correlation over `columns`, in that order, with an exact 1.0 diagonal."""
    corr = df[columns].corr(method="pearson")
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=list(columns), columns=list(columns))


def zscores(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Population (ddof=0) z-scores; a zero-variance column maps to 0.0 (NaN stays NaN)."""
    out = {}
    for col in columns:
        s = df[col].astype(float)
        std = s.std(ddof=0)
        out[col] = (s - s.mean()) / std if std > 0 else s * 0.0
    return pd.DataFrame(out, index=df.index)
