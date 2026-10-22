"""Unit tests for the hand-written Hopfield associative memory (tp4/models/hopfield.py)."""

from __future__ import annotations

import numpy as np
import pytest

from tp4.experiments.hopfield_letters import load_alphabet
from tp4.models.hopfield import (
    FIELD_TOL,
    MAX_ITER,
    HopfieldConfig,
    check_patterns,
    energy,
    hebbian_weights,
    is_fixed_point,
    recall,
    sign_keep,
)


def _random_patterns(p: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(p, n))


def test_course_worked_example_weights_and_one_step_sync_recall():
    P = np.array([[1, 1, -1, -1], [-1, -1, 1, 1]], dtype=np.int8)
    W = hebbian_weights(P)
    expected_W = np.array(
        [
            [0.0, 0.5, -0.5, -0.5],
            [0.5, 0.0, -0.5, -0.5],
            [-0.5, -0.5, 0.0, 0.5],
            [-0.5, -0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    assert np.array_equal(W, expected_W)
    res = recall(W, np.array([1, -1, -1, -1], dtype=np.int8), HopfieldConfig(update="sync", max_iter=10))
    assert res.states.shape == (1, 4)
    assert np.array_equal(res.states[0], P[0])
    assert int(res.iterations[0]) == 1
    assert str(res.status[0]) == "converged"


def test_selected_letters_are_fixed_points_under_async_and_sync():
    letters, patterns, _ = load_alphabet("datos/letras_5x5.json")
    idx = [letters.index(ch) for ch in ("G", "R", "T", "V")]
    P = patterns[idx]
    W = hebbian_weights(P)
    assert np.all(is_fixed_point(W, P))
    for upd in ("async", "sync"):
        rng = np.random.default_rng(0) if upd == "async" else None
        res = recall(W, P, HopfieldConfig(update=upd, max_iter=50), rng=rng)
        assert np.array_equal(res.states, P)
        assert np.all(res.iterations == 0)
        assert np.all(res.status == "converged")


@pytest.mark.parametrize("update", ["async", "sync"])
def test_random_stored_patterns_are_fixed_points(update: str):
    P = _random_patterns(4, 200, seed=0)
    W = hebbian_weights(P)
    assert np.all(is_fixed_point(W, P))
    rng = np.random.default_rng(7) if update == "async" else None
    res = recall(W, P, HopfieldConfig(update=update, max_iter=50), rng=rng)
    assert np.array_equal(res.states, P)
    assert np.all(res.iterations == 0)
    assert np.all(res.status == "converged")
    for e_seq in res.energies:
        assert len(e_seq) == 1


def test_async_energy_never_increases_per_neuron_update_or_per_iteration():
    P = _random_patterns(5, 100, seed=10)
    Q = _random_patterns(30, 100, seed=11)
    W = hebbian_weights(P)
    res = recall(
        W,
        Q,
        HopfieldConfig(update="async", max_iter=100),
        rng=np.random.default_rng(12),
        trace=True,
    )
    assert res.update_energies is not None
    assert len(res.update_energies) == 30
    final_energies = energy(W, res.states)
    for row in range(30):
        u_e = res.update_energies[row]
        it_e = res.energies[row]
        assert np.all(np.diff(u_e) <= 1e-12)
        assert np.all(np.diff(it_e) <= 1e-12)
        assert u_e[0] == pytest.approx(it_e[0], abs=1e-9)
        assert u_e[-1] == pytest.approx(final_energies[row], abs=1e-9)
        assert it_e[-1] == pytest.approx(final_energies[row], abs=1e-9)


def test_two_neuron_sync_cycle_and_async_convergence():
    xi = np.array([[1, -1]], dtype=np.int8)
    W = hebbian_weights(xi)
    q = np.array([1, 1], dtype=np.int8)
    res_sync = recall(W, q, HopfieldConfig(update="sync", max_iter=20))
    assert str(res_sync.status[0]) == "cycle"
    assert int(res_sync.iterations[0]) == 2
    assert np.array_equal(res_sync.states[0], q)

    for seed in range(10):
        res_async = recall(W, q, HopfieldConfig(update="async", max_iter=20), rng=np.random.default_rng(seed))
        assert str(res_async.status[0]) == "converged"
        assert np.array_equal(res_async.states[0], xi[0]) or np.array_equal(res_async.states[0], -xi[0])


def test_batch_recall_matches_single_query_recall_row_by_row():
    P = _random_patterns(4, 64, seed=1)
    Q = _random_patterns(8, 64, seed=2)
    W = hebbian_weights(P)
    cfg = HopfieldConfig(update="async", max_iter=50)
    batch_res = recall(W, Q, cfg, rng=np.random.default_rng(3), trace=True)
    assert batch_res.update_energies is not None

    for i in range(8):
        single_res = recall(W, Q[i], cfg, rng=np.random.default_rng(3), trace=True)
        assert single_res.update_energies is not None
        assert np.array_equal(batch_res.states[i], single_res.states[0])
        assert int(batch_res.iterations[i]) == int(single_res.iterations[0])
        assert str(batch_res.status[i]) == str(single_res.status[0])
        assert np.allclose(batch_res.energies[i], single_res.energies[0], atol=1e-12, rtol=0)
        assert np.allclose(batch_res.update_energies[i], single_res.update_energies[0], atol=1e-12, rtol=0)


def test_sign_keep_respects_field_tolerance():
    h = np.array([0.0, 2.0, -1e-12, -3.0])
    s = np.array([-1, -1, 1, 1], dtype=np.int8)
    out = sign_keep(h, s)
    assert out.dtype == np.int8
    assert out.tolist() == [-1, 1, 1, -1]
    assert FIELD_TOL == 1e-9


def test_max_iter_stops_and_records_status():
    xi = np.array([[1, -1]], dtype=np.int8)
    W = hebbian_weights(xi)
    q = np.array([1, 1], dtype=np.int8)
    res_sync = recall(W, q, HopfieldConfig(update="sync", max_iter=1))
    assert str(res_sync.status[0]) == "max_iter"
    assert int(res_sync.iterations[0]) == 1

    res_async = recall(W, q, HopfieldConfig(update="async", max_iter=1), rng=np.random.default_rng(0))
    assert str(res_async.status[0]) == "max_iter"
    assert int(res_async.iterations[0]) == 1


def test_validation_errors():
    P = _random_patterns(3, 10, seed=0)
    W = hebbian_weights(P)

    for bad_p in (
        np.array([1, -1, 1], dtype=np.int8),
        np.empty((0, 4), dtype=np.int8),
        np.ones((2, 1), dtype=np.int8),
        np.array([[1, 0, -1, 1]], dtype=np.int8),
        np.array([[1, 2, -1, 1]], dtype=np.int8),
        np.array([[True, False, True, False]]),
    ):
        with pytest.raises(ValueError):
            check_patterns(bad_p)

    W_nonsym = W.copy()
    W_nonsym[0, 1] += 0.1
    W_diag = W.copy()
    W_diag[0, 0] = 0.5
    W_nan = W.copy()
    W_nan[0, 1] = W_nan[1, 0] = np.nan

    for bad_W in (np.ones((3, 4)), np.zeros((1, 1)), W_nonsym, W_diag, W_nan):
        with pytest.raises(ValueError):
            recall(bad_W, P[0], HopfieldConfig(update="sync"))

    for bad_q in (
        np.ones(9, dtype=np.int8),
        np.empty((0, 10), dtype=np.int8),
        np.zeros((1, 10), dtype=np.int8),
    ):
        with pytest.raises(ValueError):
            recall(W, bad_q, HopfieldConfig(update="sync"))

    with pytest.raises(ValueError, match="unknown update"):
        recall(W, P[0], HopfieldConfig(update="greedy"))

    for bad_iter in (0, -1, MAX_ITER + 1, True):
        with pytest.raises(ValueError, match="max_iter"):
            recall(W, P[0], HopfieldConfig(update="sync", max_iter=bad_iter))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Generator"):
        recall(W, P[0], HopfieldConfig(update="async"), rng=None)

    with pytest.raises(ValueError, match="trace"):
        recall(W, P[0], HopfieldConfig(update="sync"), trace=True)
