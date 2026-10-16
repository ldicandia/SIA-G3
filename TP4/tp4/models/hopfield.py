"""Discrete Hopfield associative memory written by hand in numpy (no ML library).

Course conventions (clases/4.2 - Modelo de Hopfield.pdf):
- States in {-1, +1} (slides 6, 9).
- Local field h_i = sum_{j != i} w_ij S_j (slide 14) and update rule S_i = sign(h_i) with
  S_i unchanged when h_i == 0 (slide 15; implemented with |h_i| <= FIELD_TOL).
- Hebbian weights w_ij = (1 / N) * sum_{mu=1}^p xi_i^mu xi_j^mu for i != j and w_ii = 0,
  w_ji = w_ij (slides 18, 20). Slide 28 writes (1/N) K K^T - I, which zeroes the diagonal
  only when p = N; the explicit zero diagonal of slide 20 is used here.
- Synchronous updates S(t+1) = sign(W S(t)) (slide 22, worked example slides 25-30) with
  period-2 cycle detection (slide 38).
- Energy H(S) = -1/2 * sum_{i,j} w_ij S_i S_j (slides 34-36), non-increasing under
  asynchronous updates (flipping S_i with local field h_i changes H by 2 * S_i_old * h_i < 0).
- Asynchronous updates draw one random permutation of the N neurons per sweep from the
  seeded Generator and share it across every active row of a batch (INFRA-04), so a row's
  trajectory is identical whether recalled alone or in a batch.
- `iterations` counts sweeps that changed at least one neuron; the final no-change sweep
  is the convergence check and is not counted.

Backend: `recall(..., xp=np)` and `energy(..., xp=np)` run the sweep loops and the per-sweep batch
energies on the array module `xp` (numpy by default, cupy when the runner resolved it). The async
step has no data-dependent host branch (flips are masked with `where`), and the state, the energies
and the optional trace come back to the host once per sweep, where all the bookkeeping (active
set, iterations, status, cycle detection) stays numpy. Random permutations are always drawn on
the host Generator, once per sweep, so the same seed gives the same orders on any backend.
Weight construction and validation (`train`, `hebbian_weights`, `pseudoinverse_weights`,
`_check_weights`, `check_patterns`, `is_fixed_point`, `sign_keep`) stay on the host: they are
one-shot and tiny, and they rely on exact host checks (symmetry, zero diagonal). Hebb weights are
sums of +-1 products and therefore exact on any backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend import to_numpy

FIELD_TOL = 1e-9
MAX_ITER = 1000
RULES = ("hebb", "pseudoinverse")
UPDATES = ("async", "sync")
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


def pseudoinverse_weights(patterns) -> np.ndarray:
    """Projection (pseudoinverse) weight matrix W = K (K^T K)^{-1} K^T with zero diagonal.

    Every stored pattern satisfies W' xi = xi before the diagonal is removed. With the
    diagonal removed, h_i = (1 - d_i) xi_i with d_i = W'_ii < 1, so stored patterns stay
    fixed points even when correlated. This is a different learning rule from the course's
    Hebb rule, used in plan 04-03 to show that correlation, not pattern count, breaks Hebb
    on the QR codes.
    """
    P = check_patterns(patterns).astype(np.float64)
    p = P.shape[0]
    K = P.T
    C = K.T @ K
    if int(np.linalg.matrix_rank(C)) < p:
        raise ValueError("patterns are linearly dependent; pseudoinverse_weights requires full row rank")
    W = K @ np.linalg.solve(C, K.T)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def train(patterns, rule: str = "hebb") -> np.ndarray:
    """Build the synaptic weight matrix W for `patterns` using `rule`."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; allowed: {list(RULES)}")
    if rule == "hebb":
        return hebbian_weights(patterns)
    return pseudoinverse_weights(patterns)


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


def energy(W, states, xp=np) -> float | np.ndarray:
    """Hopfield energy H = -1/2 s^T W s (float for 1-D state, float64 1-D array for 2-D batch).

    The arithmetic runs on `xp`; the result is always a host float / numpy array.
    """
    W_arr = xp.asarray(W, dtype=np.float64)
    s_arr = xp.asarray(states, dtype=np.float64)
    if s_arr.ndim == 1:
        return float(-0.5 * float(s_arr @ W_arr @ s_arr))
    if s_arr.ndim == 2:
        return to_numpy(-0.5 * xp.sum((s_arr @ W_arr) * s_arr, axis=1))
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
    xp=np,
) -> RecallResult:
    """Run Hopfield associative recall on `queries` (1-D single query or 2-D batch).

    The sweeps and the batch energies run on `xp`; the returned arrays are always host numpy.
    """
    W_arr = _check_weights(W)
    n = W_arr.shape[0]

    q_raw = np.asarray(queries)
    if q_raw.ndim == 1:
        q_raw = q_raw.reshape(1, -1)
    S = check_patterns(q_raw)
    if S.shape[1] != n:
        raise ValueError(f"queries width {S.shape[1]} does not match W size {n}")

    if config.rule not in RULES:
        raise ValueError(f"unknown rule {config.rule!r}; allowed: {list(RULES)}")
    if config.update not in UPDATES:
        raise ValueError(f"unknown update {config.update!r}; allowed: {list(UPDATES)}")
    if not (_is_int(config.max_iter) and 1 <= config.max_iter <= MAX_ITER):
        raise ValueError(f"max_iter must be an integer in [1, {MAX_ITER}], got {config.max_iter!r}")
    if config.update == "async" and not isinstance(rng, np.random.Generator):
        raise ValueError("async update requires a numpy.random.Generator rng")
    if trace and config.update != "async":
        raise ValueError("trace=True is only supported for async update")

    b = S.shape[0]
    iterations = np.zeros(b, dtype=int)
    status = np.full(b, "max_iter", dtype=object)
    W_dev = xp.asarray(W_arr)  # one upload per call; bookkeeping below stays on the host
    init_e = np.asarray(energy(W_dev, S, xp=xp), dtype=np.float64)
    energies: list[list[float]] = [[float(init_e[row])] for row in range(b)]
    update_traces: list[list[float]] | None = [[float(init_e[row])] for row in range(b)] if trace else None
    cur_e = init_e.copy()
    active = np.ones(b, dtype=bool)

    if config.update == "async":
        for t in range(1, int(config.max_iter) + 1):
            if not np.any(active):
                break
            act_idx = np.flatnonzero(active)
            S_prev = S[act_idx].copy()
            perm = rng.permutation(n)  # type: ignore[union-attr]
            S_act = xp.asarray(S[act_idx].astype(np.float64))
            e_act = xp.asarray(cur_e[act_idx].copy())
            if update_traces is not None:
                step_e = xp.empty((act_idx.size, n), dtype=np.float64)
            for step_pos, i in enumerate(perm):
                h = S_act @ W_dev[int(i)]
                old_s = S_act[:, int(i)]
                new_s = xp.where(h > FIELD_TOL, 1.0, xp.where(h < -FIELD_TOL, -1.0, old_s))
                flipped = new_s != old_s
                # No host branch on `flipped`: the energy change is masked, and it is computed
                # before the column is overwritten (old_s is a view of it).
                e_act = e_act + xp.where(flipped, 2.0 * old_s * h, 0.0)
                S_act[:, int(i)] = new_s
                if update_traces is not None:
                    step_e[:, step_pos] = e_act
            S_new = to_numpy(S_act).astype(np.int8)  # one device-to-host copy per sweep
            S[act_idx] = S_new
            e_act_host = to_numpy(e_act)
            cur_e[act_idx] = e_act_host
            if update_traces is not None:
                step_e_host = to_numpy(step_e)
                for local_pos, row in enumerate(act_idx):
                    update_traces[int(row)].extend(step_e_host[local_pos].tolist())

            changed = np.any(S_new != S_prev, axis=1)
            for row in act_idx[~changed]:
                status[int(row)] = "converged"
                iterations[int(row)] = t - 1
                active[int(row)] = False

            changed_rows = act_idx[changed]
            if changed_rows.size > 0:
                row_energies = np.asarray(energy(W_dev, S[changed_rows], xp=xp), dtype=np.float64)
                cur_e[changed_rows] = row_energies
                for row, e_val in zip(changed_rows, row_energies, strict=True):
                    energies[int(row)].append(float(e_val))
                    iterations[int(row)] = t
    else:
        S_prev2 = S.copy()
        for t in range(1, int(config.max_iter) + 1):
            if not np.any(active):
                break
            act_idx = np.flatnonzero(active)
            S_cur = S[act_idx]
            h = to_numpy(xp.asarray(S_cur.astype(np.float64)) @ W_dev)
            S_new = sign_keep(h, S_cur)
            same_as_cur = np.all(S_new == S_cur, axis=1)
            same_as_prev2 = (t >= 2) & (~same_as_cur) & np.all(S_new == S_prev2[act_idx], axis=1)

            S_prev2[act_idx] = S_cur
            S[act_idx] = S_new

            for row in act_idx[same_as_cur]:
                status[int(row)] = "converged"
                iterations[int(row)] = t - 1
                active[int(row)] = False

            changed_mask = ~same_as_cur
            changed_rows = act_idx[changed_mask]
            if changed_rows.size > 0:
                row_energies = np.asarray(energy(W_dev, S[changed_rows], xp=xp), dtype=np.float64)
                for row, e_val in zip(changed_rows, row_energies, strict=True):
                    energies[int(row)].append(float(e_val))
                    iterations[int(row)] = t

            for row in act_idx[same_as_prev2]:
                status[int(row)] = "cycle"
                active[int(row)] = False

    return RecallResult(
        states=S,
        iterations=iterations,
        status=np.asarray(status, dtype=str),
        energies=[np.asarray(e_list, dtype=np.float64) for e_list in energies],
        update_energies=(
            [np.asarray(u_list, dtype=np.float64) for u_list in update_traces]
            if update_traces is not None
            else None
        ),
    )
