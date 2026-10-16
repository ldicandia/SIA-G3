"""Comparison of two maps trained on the same creators: region agreement and layout agreement.

numpy/pandas only (no sklearn.metrics): the adjusted Rand index is written out from the
contingency counts so the comparison stays inspectable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def contingency_table(a, b) -> pd.DataFrame:
    """Counts of samples per (label in `a`, label in `b`); sorted unique labels as index and columns."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"label vectors must be 1-D and the same length, got {a.shape} and {b.shape}")
    rows = np.unique(a)
    cols = np.unique(b)
    counts = np.zeros((rows.size, cols.size), dtype=int)
    ri = np.searchsorted(rows, a)
    ci = np.searchsorted(cols, b)
    np.add.at(counts, (ri, ci), 1)
    return pd.DataFrame(counts, index=rows, columns=cols)


def _pairs(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    return float((counts * (counts - 1.0) / 2.0).sum())


def adjusted_rand_index(a, b) -> float:
    """Hubert-Arabie adjusted Rand index of two partitions of the same samples.

    1.0 for identical partitions (up to relabeling), about 0 for independent ones, negative when
    they agree less than chance. When both partitions are trivial (the index is 0/0) the result is
    1.0 if they are identical up to relabeling and 0.0 otherwise.
    """
    table = contingency_table(a, b).to_numpy()
    n = int(table.sum())
    if n < 2:
        return 1.0
    total_pairs = n * (n - 1) / 2.0
    sum_ij = _pairs(table)
    sum_a = _pairs(table.sum(axis=1))
    sum_b = _pairs(table.sum(axis=0))
    expected = sum_a * sum_b / total_pairs
    maximum = (sum_a + sum_b) / 2.0
    if np.isclose(maximum, expected, rtol=1e-12, atol=1e-12):
        identical = bool(
            table.shape[0] == table.shape[1] and (np.count_nonzero(table, axis=1) == 1).all()
            and (np.count_nonzero(table, axis=0) == 1).all()
        )
        return 1.0 if identical else 0.0
    return float((sum_ij - expected) / (maximum - expected))


def grid_distance_agreement(dist_a, bmu_a, dist_b, bmu_b) -> float:
    """Pearson r, over creator pairs, of the BMU grid distances on map A against map B.

    `dist_*` are the grids' neuron-to-neuron distance matrices and `bmu_*` each creator's winning
    neuron. NaN when fewer than 3 creators or when either side is constant.
    """
    bmu_a = np.asarray(bmu_a, dtype=int)
    bmu_b = np.asarray(bmu_b, dtype=int)
    n = bmu_a.size
    if n < 3 or bmu_b.size != n:
        return float("nan")
    da = np.asarray(dist_a, dtype=float)[bmu_a][:, bmu_a]
    db = np.asarray(dist_b, dtype=float)[bmu_b][:, bmu_b]
    i, j = np.triu_indices(n, k=1)
    va, vb = da[i, j], db[i, j]
    if va.std() == 0.0 or vb.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(va, vb)[0, 1])
