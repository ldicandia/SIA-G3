"""Pure PCA helpers around scikit-learn's PCA (the one library model allowed in TP4).

Standardization is z-scores with population std (ddof=0, StandardScaler
semantics). Components are sign-oriented so results do not depend on sklearn's
version-dependent sign convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..backend import to_numpy
from ..errors import DataError


@dataclass(frozen=True)
class PcaResult:
    features: tuple[str, ...]
    components: np.ndarray  # (k, n_features), unit rows, sign-oriented
    explained_variance: np.ndarray  # sklearn convention (ddof=1)
    explained_variance_ratio: np.ndarray
    scores: np.ndarray  # (n_samples, k) = Z @ components.T
    n_samples: int

    @property
    def cumulative_ratio(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the ddof=0 covariance (= correlation matrix when standardized)."""
        return self.explained_variance * (self.n_samples - 1) / self.n_samples


def standardize(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Z, mean, std) with ddof=0. Raises DataError naming every NaN or constant column."""
    values = X.to_numpy(dtype=float)
    nan_cols = [c for c, bad in zip(X.columns, np.isnan(values).any(axis=0)) if bad]
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    const_cols = [c for c, s, m in zip(X.columns, std, mean) if not np.isnan(m) and not s > 0]
    if nan_cols or const_cols:
        parts = []
        if nan_cols:
            parts.append(f"NaN values in {nan_cols}")
        if const_cols:
            parts.append(f"zero variance in {const_cols}")
        raise DataError("cannot standardize: " + "; ".join(parts))
    return (values - mean) / std, mean, std


def center(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(X - mean, mean). Raises DataError on NaN."""
    values = X.to_numpy(dtype=float)
    nan_cols = [c for c, bad in zip(X.columns, np.isnan(values).any(axis=0)) if bad]
    if nan_cols:
        raise DataError(f"cannot center: NaN values in {nan_cols}")
    mean = values.mean(axis=0)
    return values - mean, mean


def orient_components(components: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip each component (and its score column) so its largest-|coef| entry is positive.

    Ties go to the lowest feature index (np.argmax returns the first maximum).
    """
    comps = np.array(components, dtype=float, copy=True)
    sc = np.array(scores, dtype=float, copy=True)
    for i in range(comps.shape[0]):
        if comps[i, int(np.argmax(np.abs(comps[i])))] < 0:
            comps[i] = -comps[i]
            sc[:, i] = -sc[:, i]
    return comps, sc


def fit_pca(Z: np.ndarray, features: Sequence[str], n_components: int | None = None) -> PcaResult:
    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(Z)
    components, scores = orient_components(model.components_, Z @ model.components_.T)
    return PcaResult(
        features=tuple(features),
        components=components,
        explained_variance=model.explained_variance_.copy(),
        explained_variance_ratio=model.explained_variance_ratio_.copy(),
        scores=scores,
        n_samples=int(Z.shape[0]),
    )


def eigen_crosscheck(Z: np.ndarray, explained_variance_ratio: np.ndarray, xp: ModuleType) -> float:
    """Max |difference| between sklearn's ratios and an eigen-decomposition on backend `xp`.

    This is the course's view of PCA: eigenvalues of the covariance `Z.T @ Z / n`
    (the correlation matrix when `Z` is standardized), normalized to ratios.
    """
    Zx = xp.asarray(Z)
    cov = Zx.T @ Zx / Zx.shape[0]
    eigvals = np.sort(to_numpy(xp.linalg.eigvalsh(cov)))[::-1]
    ratios = eigvals / eigvals.sum()
    expected = np.asarray(explained_variance_ratio, dtype=float)
    return float(np.max(np.abs(ratios[: expected.size] - expected)))


def components_for_threshold(cumulative: np.ndarray, threshold: float) -> int:
    """Smallest k (1-based) with cumulative[k-1] >= threshold (1e-12 slack); full count if never."""
    cum = np.asarray(cumulative, dtype=float)
    hits = np.flatnonzero(cum >= threshold - 1e-12)
    return int(hits[0]) + 1 if hits.size else int(cum.size)


def kaiser_count(result: PcaResult) -> int:
    """Number of correlation-matrix eigenvalues greater than 1."""
    return int((result.eigenvalues > 1.0).sum())


def loadings_frame(result: PcaResult, n_pcs: int | None = None) -> pd.DataFrame:
    """Oriented eigenvector coefficients: one row per feature (CSV order), columns PC1..PCk."""
    k = result.components.shape[0] if n_pcs is None else min(n_pcs, result.components.shape[0])
    return pd.DataFrame(
        result.components[:k].T,
        index=list(result.features),
        columns=[f"PC{i + 1}" for i in range(k)],
    )


def top_loadings(result: PcaResult, pc: int, n: int) -> list[tuple[str, float]]:
    """The n features with the largest |coefficient| on component `pc` (0-based), ties by name."""
    coefs = result.components[pc]
    order = sorted(range(len(coefs)), key=lambda i: (-abs(float(coefs[i])), result.features[i]))
    return [(result.features[i], float(coefs[i])) for i in order[:n]]
