"""The models and experiments run on `ctx.backend.xp`; a numpy-delegating fake cupy proves the xp
code paths give the same results as numpy without needing CUDA."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tp4.cli import main
from tp4.models.oja import OjaConfig, train_oja


def _fake_cupy(devices: int = 1) -> types.ModuleType:
    """A stand-in `cupy` that delegates every array function to numpy."""
    fake = types.ModuleType("cupy")
    fake.__getattr__ = lambda name: getattr(np, name)  # PEP 562 module __getattr__
    fake.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: devices))
    return fake


EIGENVALUES = np.array([5.0, 2.0, 1.0, 0.5, 0.25])


def _anisotropic(n: int = 200, seed: int = 0):
    """Centered Gaussian with known covariance spectrum in a random orthonormal basis."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(EIGENVALUES.size, EIGENVALUES.size)))
    X = rng.normal(size=(n, EIGENVALUES.size)) * np.sqrt(EIGENVALUES) @ basis.T
    return X - X.mean(axis=0)


def _oja_cfg(**kw):
    base = dict(epochs=30, eta0=1e-3, eta_min=1e-5, eta_decay="constant")
    base.update(kw)
    return OjaConfig(**base)


def _assert_oja_equal(a, b):
    assert isinstance(b.weights, np.ndarray) and isinstance(b.initial_weights, np.ndarray)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.initial_weights, b.initial_weights)
    assert set(a.history) == set(b.history)
    for key in a.history:
        assert isinstance(b.history[key], np.ndarray)
        np.testing.assert_array_equal(a.history[key], b.history[key])
    assert a.diverged == b.diverged and a.divergence_epoch == b.divergence_epoch


# --------------------------------------------------------------------------- Oja


@pytest.mark.parametrize("decay", ["constant", "linear"])
def test_oja_fake_xp_matches_numpy(decay):
    X = _anisotropic()
    reference = np.linalg.eigh(X.T @ X)[1][:, -1]
    cfg = _oja_cfg(eta_decay=decay)
    a = train_oja(X, cfg, np.random.default_rng(42), reference=reference)
    b = train_oja(X, cfg, np.random.default_rng(42), reference=reference, xp=_fake_cupy())
    _assert_oja_equal(a, b)
    assert a.diverged is False


def test_oja_divergence_is_flagged_identically_under_xp():
    X = _anisotropic()
    cfg = _oja_cfg(eta0=5.0, eta_min=1e-5)
    a = train_oja(X, cfg, np.random.default_rng(3))
    b = train_oja(X, cfg, np.random.default_rng(3), xp=_fake_cupy())
    assert a.diverged is True and b.diverged is True
    assert a.divergence_epoch == b.divergence_epoch
    assert np.isnan(b.weights).all()
    _assert_oja_equal(a, b)


def test_oja_rng_state_is_the_same_after_training_with_either_xp():
    X = _anisotropic()
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    train_oja(X, _oja_cfg(), rng_a)
    train_oja(X, _oja_cfg(), rng_b, xp=_fake_cupy())
    assert rng_a.random() == rng_b.random()


# --------------------------------------------------------------------------- CLI helpers


def _run_cli(config: Path, out_root: Path) -> Path:
    assert main(["run", str(config), "--out-root", str(out_root)]) == 0
    runs = [p for p in out_root.iterdir() if p.is_dir()]
    assert len(runs) == 1
    return runs[0]


def _compare_runs(numpy_dir: Path, cupy_dir: Path, csvs: list[str] | None) -> dict:
    a = json.loads((numpy_dir / "summary.json").read_text(encoding="utf-8"))
    b = json.loads((cupy_dir / "summary.json").read_text(encoding="utf-8"))
    assert a.pop("backend_resolved") == "numpy"
    assert b.pop("backend_resolved") == "cupy"
    assert a == b
    if csvs is None:  # every csv the numpy run wrote
        csvs = sorted(p.name for p in numpy_dir.glob("*.csv"))
        assert csvs
    for name in csvs:
        assert (numpy_dir / name).read_bytes() == (cupy_dir / name).read_bytes(), name
    return json.loads((cupy_dir / "run.json").read_text(encoding="utf-8"))


def _cli_numpy_vs_fake(write_config, tmp_path, monkeypatch, base, csvs, **overrides):
    numpy_cfg = write_config(base, backend="numpy", plots=False, **overrides)
    cupy_cfg = write_config(base, backend="cupy", plots=False, **overrides)
    numpy_dir = _run_cli(numpy_cfg, tmp_path / "numpy")
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    cupy_dir = _run_cli(cupy_cfg, tmp_path / "cupy")
    return _compare_runs(numpy_dir, cupy_dir, csvs)


def test_cli_oja_fake_cupy_equals_numpy(write_config, tmp_path, monkeypatch):
    record = _cli_numpy_vs_fake(
        write_config, tmp_path, monkeypatch, "configs/oja.json",
        ["ranking.csv", "history.csv", "weights_vs_pc1.csv"],
    )
    assert record["backend"]["resolved"] == "cupy" and record["backend"]["fallback_reason"] is None


def test_cli_campaigns_fake_cupy_equals_numpy(write_config, tmp_path, monkeypatch):
    _cli_numpy_vs_fake(
        write_config, tmp_path, monkeypatch, "configs/campaigns.json", ["evidence.csv", "picks.csv"]
    )


# --------------------------------------------------------------------------- Hopfield


def _patterns(p: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice([-1, 1], size=(p, n)).astype(np.int8)


def _noisy_queries(P: np.ndarray, flips: list[int], seed: int) -> np.ndarray:
    """One query per entry of `flips`, cycling over the stored patterns, with that many bits flipped."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, n_flip in enumerate(flips):
        q = P[k % P.shape[0]].copy()
        idx = rng.choice(P.shape[1], size=n_flip, replace=False)
        q[idx] = -q[idx]
        rows.append(q)
    return np.array(rows, dtype=np.int8)


def _assert_recall_equal(a, b):
    np.testing.assert_array_equal(a.states, b.states)
    assert isinstance(b.states, np.ndarray) and b.states.dtype == np.int8
    np.testing.assert_array_equal(a.iterations, b.iterations)
    assert a.status.tolist() == b.status.tolist()
    assert len(a.energies) == len(b.energies)
    for ea, eb in zip(a.energies, b.energies, strict=True):
        assert isinstance(eb, np.ndarray)
        np.testing.assert_array_equal(ea, eb)
    assert (a.update_energies is None) == (b.update_energies is None)
    if a.update_energies is not None:
        for ea, eb in zip(a.update_energies, b.update_energies, strict=True):
            np.testing.assert_array_equal(ea, eb)


@pytest.mark.parametrize("rule", ["hebb", "pseudoinverse"])
@pytest.mark.parametrize("update,trace", [("async", False), ("async", True), ("sync", False)])
def test_recall_fake_xp_matches_numpy(rule, update, trace):
    from tp4.models.hopfield import HopfieldConfig, recall, train

    P = _patterns(4, 60, seed=1)
    W = train(P, rule)
    Q = _noisy_queries(P, [0, 3, 8, 15, 24, 30, 6, 12], seed=2)
    cfg = HopfieldConfig(rule=rule, update=update, max_iter=50)
    a = recall(W, Q, cfg, np.random.default_rng(5), trace=trace)
    b = recall(W, Q, cfg, np.random.default_rng(5), trace=trace, xp=_fake_cupy())
    _assert_recall_equal(a, b)


def test_recall_active_set_shrinks_mid_run_and_still_matches():
    from tp4.models.hopfield import HopfieldConfig, recall, train

    P = _patterns(3, 80, seed=11)
    W = train(P, "hebb")
    Q = _noisy_queries(P, [0, 1, 5, 12, 22, 30, 36], seed=12)
    cfg = HopfieldConfig(rule="hebb", update="async", max_iter=60)
    a = recall(W, Q, cfg, np.random.default_rng(9), trace=True)
    b = recall(W, Q, cfg, np.random.default_rng(9), trace=True, xp=_fake_cupy())
    assert len(set(a.iterations.tolist())) > 1  # rows converge at different sweeps
    _assert_recall_equal(a, b)


def test_recall_sync_cycle_and_max_iter_statuses_match():
    from tp4.models.hopfield import HopfieldConfig, recall, train

    # Two neurons with W = [[0, -1], [-1, 0]]: synchronous (+1, +1) flips to (-1, -1) and back.
    W = np.array([[0.0, -1.0], [-1.0, 0.0]])
    q = np.array([[1, 1]], dtype=np.int8)
    cfg = HopfieldConfig(rule="hebb", update="sync", max_iter=20)
    a = recall(W, q, cfg)
    b = recall(W, q, cfg, xp=_fake_cupy())
    assert a.status.tolist() == b.status.tolist() == ["cycle"]
    _assert_recall_equal(a, b)

    P = _patterns(4, 60, seed=3)
    Wh = train(P, "hebb")
    Q = _noisy_queries(P, [20, 25, 28, 30], seed=4)
    cfg = HopfieldConfig(rule="hebb", update="async", max_iter=1)
    a = recall(Wh, Q, cfg, np.random.default_rng(0))
    b = recall(Wh, Q, cfg, np.random.default_rng(0), xp=_fake_cupy())
    assert "max_iter" in a.status.tolist()
    _assert_recall_equal(a, b)


def test_recall_rng_state_is_the_same_with_either_xp():
    from tp4.models.hopfield import HopfieldConfig, recall, train

    P = _patterns(3, 40, seed=6)
    W = train(P, "hebb")
    Q = _noisy_queries(P, [4, 8, 12], seed=7)
    cfg = HopfieldConfig(rule="hebb", update="async", max_iter=30)
    rng_a, rng_b = np.random.default_rng(21), np.random.default_rng(21)
    recall(W, Q, cfg, rng_a)
    recall(W, Q, cfg, rng_b, xp=_fake_cupy())
    assert rng_a.random() == rng_b.random()


def test_energy_xp_matches_numpy_and_returns_host_values():
    from tp4.models.hopfield import energy, train

    P = _patterns(3, 30, seed=8)
    W = train(P, "hebb")
    fake = _fake_cupy()
    e1 = energy(W, P[0].astype(float), xp=fake)
    assert isinstance(e1, float) and e1 == energy(W, P[0].astype(float))
    e2 = energy(W, P, xp=fake)
    assert isinstance(e2, np.ndarray) and e2.dtype == np.float64
    np.testing.assert_array_equal(e2, energy(W, P))


# --------------------------------------------------------------------------- SOM


class _DeviceLike(np.ndarray):
    """A numpy subclass that mimics a device array: `.get()` exists and argmin picks the LAST tie."""

    def get(self):
        return np.asarray(self)

    def argmin(self, *args, **kwargs):
        flipped = np.asarray(self)[::-1]
        return len(flipped) - 1 - int(flipped.argmin())


def test_best_matching_unit_ties_go_to_the_lowest_index_whatever_the_array_type():
    from tp4.models.som import best_matching_unit

    W = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0]])
    x = np.array([0.1, -0.1])
    assert best_matching_unit(W, x) == 1
    dev = W.view(_DeviceLike)
    assert best_matching_unit(dev, x) == 1
    assert type(best_matching_unit(dev, x)) is int


class _ReversedSumDevice:
    """Duck-typed device array whose row sums add the columns in reverse order, like a GPU
    reduction tree that associates differently from numpy's."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=float)

    def __sub__(self, other):
        return _ReversedSumDevice(self.arr - np.asarray(other))

    def __pow__(self, p):
        return _ReversedSumDevice(self.arr**p)

    def sum(self, axis):
        assert axis == 1
        out = np.zeros(self.arr.shape[0])
        for k in range(self.arr.shape[1] - 1, -1, -1):
            out = out + self.arr[:, k]
        return _ReversedSumDevice(out)

    def get(self):
        return self.arr


def test_best_matching_unit_near_ties_follow_the_numpy_decision_on_a_device_array():
    from tp4.models.som import best_matching_unit

    # Row 0: squares 1e16, 1, 1. Row 1: 1e16, 0, 0. In numpy's order both sums are exactly 1e16
    # (a tie, so index 0); added in reverse the first becomes 1e16 + 2 and a device argmin picks 1.
    W = np.array([[1e8, 1.0, 1.0], [1e8, 0.0, 0.0]])
    x = np.zeros(3)
    assert best_matching_unit(W, x) == 0
    dev = _ReversedSumDevice(W)
    assert int(np.argmin(((dev - x) ** 2).sum(axis=1).get())) == 1  # the raw device decision differs
    assert best_matching_unit(dev, x) == 0  # the near-tie is re-decided with numpy's own sum


def test_train_som_duplicate_initial_prototypes_match_under_fake_xp():
    from tp4.models.som import SomConfig, train_som

    X = np.random.default_rng(0).normal(size=(10, 4))
    cfg = SomConfig(rows=4, cols=4, init="sample", epochs=5)  # 16 neurons > 10 samples: duplicates
    a = train_som(X, cfg, np.random.default_rng(1))
    b = train_som(X, cfg, np.random.default_rng(1), xp=_fake_cupy())
    np.testing.assert_array_equal(a.weights, b.weights)


# --------------------------------------------------------------------------- CLI: Hopfield and SOM


def test_cli_hopfield_letters_fake_cupy_equals_numpy(write_config, tmp_path, monkeypatch):
    _cli_numpy_vs_fake(
        write_config, tmp_path, monkeypatch, "configs/hopfield_letters.json", None,
        **{"params.trials": 6, "params.capacity_subsets": 5},
    )


def test_cli_hopfield_qr_fake_cupy_equals_numpy(write_config, tmp_path, monkeypatch):
    _cli_numpy_vs_fake(
        write_config, tmp_path, monkeypatch, "configs/hopfield_qr.json", None,
        **{
            "params.trials": 4,
            "params.noise_levels": [0.0, 0.05, 0.3, 0.45, 1.0],
            "params.torn_corner_sizes": [8, 20],
            "params.occlusion_sizes": [12],
            "params.trace_levels": [0.3],
        },
    )


def test_cli_som_fake_cupy_equals_numpy(write_config, tmp_path, monkeypatch):
    _cli_numpy_vs_fake(
        write_config, tmp_path, monkeypatch, "configs/som.json", None, **{"params.epochs": 15}
    )
