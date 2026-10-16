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


def _compare_runs(numpy_dir: Path, cupy_dir: Path, csvs: list[str]) -> dict:
    a = json.loads((numpy_dir / "summary.json").read_text(encoding="utf-8"))
    b = json.loads((cupy_dir / "summary.json").read_text(encoding="utf-8"))
    assert a.pop("backend_resolved") == "numpy"
    assert b.pop("backend_resolved") == "cupy"
    assert a == b
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
