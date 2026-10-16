"""Discrete Hopfield associative memory written by hand in numpy (no ML library).

Course conventions (clases/4.2 - Modelo de Hopfield.pdf):
- States in {-1, +1} (slides 6, 9).
- Local field h_i = sum_{j != i} w_ij S_j (slide 14) and update rule S_i = sign(h_i) with
  S_i unchanged when h_i == 0 (slide 15; implemented with |h_i| <= FIELD_TOL).
- Hebbian weights w_ij = (1 / N) * sum_{mu=1}^p xi_i^mu xi_j^mu for i != j and w_ii = 0,
  w_ji = w_ij (slides 18, 20). Slide 28 writes (1/N) K K^T - I, which zeroes the diagonal
  only when p = N; the explicit zero diagonal of slide 20 is used here.
- Energy H(S) = -1/2 * sum_{i,j} w_ij S_i S_j (slides 34-36), non-increasing under
  asynchronous updates.
- Asynchronous updates draw one random permutation of the N neurons per sweep from the
  seeded Generator and share it across every active row of a batch (INFRA-04), so a row's
  trajectory is identical whether recalled alone or in a batch.
- `iterations` counts sweeps that changed at least one neuron; the final no-change sweep
  is the convergence check and is not counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIELD_TOL = 1e-9
MAX_ITER = 1000
RULES = ("hebb",)
UPDATES = ("async",)
STATUSES = ("converged", "cycle", "max_iter")


@dataclass(frozen=True)
class HopfieldConfig:
    rule: str = "hebb"
    update: str = "async"
    max_iter: int = 100


@dataclass(frozen=True)
class RecallResult:
    states: np.ndarray  # (B, N) int8
    iterations: np.ndarray  # (B,) int
    status: np.ndarray  # (B,) str from STATUSES
    energies: list[np.ndarray]  # length B; initial energy + one per changing iteration
    update_energies: list[np.ndarray] | None = None


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def check_patterns(patterns) -> np.ndarray:
    """Validate and return a C-contiguous int8 array of shape (p, N) with values in {-1, +1}."""
    arr = np.asarray(patterns)
    if arr.dtype.kind == "b" or arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError("patterns must be a numeric array with values in {-1, +1}")
    if arr.ndim != 2:
        raise ValueError(f"patterns must be 2-D (p x N), got {arr.ndim}-D")
    if arr.shape[0] < 1 or arr.shape[1] < 2:
        raise ValueError(f"patterns needs at least 1 row and 2 columns, got shape {arr.shape}")
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        raise ValueError("patterns contains non-finite values")
    if not bool(np.all((arr == -1) | (arr == 1))):
        raise ValueError("patterns must contain only -1 and +1")
    return np.ascontiguousarray(arr, dtype=np.int8).copy()


def hebbian_weights(patterns) -> np.ndarray:
    """Hebbian weight matrix W = (1/N) P^T P in float64 with an exact zero diagonal."""
    P = check_patterns(patterns).astype(np.float64)
    n = P.shape[1]
    W = (P.T @ P) / float(n)
    np.fill_diagonal(W, 0.0)
    return W


def train(patterns, rule: str = "hebb") -> np.ndarray:
    """Build the synaptic weight matrix W for `patterns` using `rule`."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; allowed: {list(RULES)}")
    return hebbian_weights(patterns)


def sign_keep(h, s) -> np.ndarray:
    """Return +1 where h > FIELD_TOL, -1 where h < -FIELD_TOL, else keep s (as int8)."""
    h_arr = np.asarray(h, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.int8)
    if h_arr.shape != s_arr.shape:
        raise ValueError(f"h and s must have the same shape, got {h_arr.shape} vs {s_arr.shape}")
    out = s_arr.copy()
    out[h_arr > FIELD_TOL] = 1
    out[h_arr < -FIELD_TOL] = -1
    return out


def energy(W, states) -> float | np.ndarray:
    """Hopfield energy H = -1/2 s^T W s (float for 1-D state, float64 1-D array for 2-D batch)."""
    W_arr = np.asarray(W, dtype=np.float64)
    s_arr = np.asarray(states, dtype=np.float64)
    if s_arr.ndim == 1:
        return float(-0.5 * float(s_arr @ W_arr @ s_arr))
    if s_arr.ndim == 2:
        return -0.5 * np.sum((s_arr @ W_arr) * s_arr, axis=1)
    raise ValueError(f"states must be 1-D or 2-D, got {s_arr.ndim}-D")


def is_fixed_point(W, patterns) -> np.ndarray:
    """Per row of `patterns`, whether every neuron stays unchanged under one step."""
    W_arr = _check_weights(W)
    P = check_patterns(patterns)
    if P.shape[1] != W_arr.shape[0]:
        raise ValueError(f"patterns width {P.shape[1]} does not match W shape {W_arr.shape}")
    h = P.astype(np.float64) @ W_arr
    return np.all(sign_keep(h, P) == P, axis=1)


def _check_weights(W) -> np.ndarray:
    W_arr = np.asarray(W, dtype=np.float64)
    if W_arr.ndim != 2 or W_arr.shape[0] != W_arr.shape[1] or W_arr.shape[0] < 2:
        raise ValueError(f"W must be a square 2-D array of size >= 2, got shape {W_arr.shape}")
    if not np.isfinite(W_arr).all():
        raise ValueError("W must contain only finite values")
    if not np.array_equal(W_arr, W_arr.T):
        raise ValueError("W must be exactly symmetric")
    if not np.all(np.diag(W_arr) == 0.0):
        raise ValueError("W must have an exactly zero diagonal")
    return W_arr


def recall(
    W,
    queries,
    config: HopfieldConfig = HopfieldConfig(),
    rng: np.random.Generator | None = None,
    *,
    trace: bool = False,
) -> RecallResult:
    """Run Hopfield associative recall on `queries` (1-D single query or 2-D batch)."""
    W_arr = _check_weights(W)
    n = W_arr.shape[0]

    q_raw = np.asarray(queries)
    if q_raw.ndim == 1:
        q_raw = q_raw.reshape(1, -1)
    S = check_patterns(q_raw)
    if S.shape[1] != n:
        raise ValueError(f"queries width {S.shape[1]} does not match W size {n}")

    if config.update not in UPDATES:
        raise ValueError(f"unknown update {config.update!r}; allowed: {list(UPDATES)}")
    if not (_is_int(config.max_iter) and 1 <= config.max_iter <= MAX_ITER):
        raise ValueError(f"max_iter must be an integer in [1, {MAX_ITER}], got {config.max_iter!r}")
    if config.update == "async" and not isinstance(rng, np.random.Generator):
        raise ValueError("async update requires a numpy.random.Generator rng")

    b = S.shape[0]
    iterations = np.zeros(b, dtype=int)
    status = np.full(b, "max_iter", dtype=object)
    init_e = np.asarray(energy(W_arr, S), dtype=np.float64)
    energies: list[list[float]] = [[float(init_e[row])] for row in range(b)]
    active = np.ones(b, dtype=bool)

    for t in range(1, int(config.max_iter) + 1):
        if not np.any(active):
            break
        act_idx = np.flatnonzero(active)
        S_prev = S[act_idx].copy()
        perm = rng.permutation(n)  # type: ignore[union-attr]
        S_act = S[act_idx].astype(np.float64)
        for i in perm:
            h = S_act @ W_arr[i]
            new_val = np.where(h > FIELD_TOL, 1.0, np.where(h < -FIELD_TOL, -1.0, S_act[:, i]))
            S_act[:, i] = new_val
        S_new = S_act.astype(np.int8)
        S[act_idx] = S_new

        changed = np.any(S_new != S_prev, axis=1)
        converged_rows = act_idx[~changed]
        for row in converged_rows:
            status[row] = "converged"
            iterations[row] = t - 1
            active[row] = False

        changed_rows = act_idx[changed]
        if changed_rows.size > 0:
            row_energies = np.asarray(energy(W_arr, S[changed_rows]), dtype=np.float64)
            for row, e_val in zip(changed_rows, row_energies, strict=True):
                energies[int(row)].append(float(e_val))
                iterations[int(row)] = t

    return RecallResult(
        states=S,
        iterations=iterations,
        status=np.asarray(status, dtype=str),
        energies=[np.asarray(e_list, dtype=np.float64) for e_list in energies],
        update_energies=None,
    )
