from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from tp4.analysis.pca import eigen_crosscheck, fit_pca, standardize
from tp4.backend import Backend, resolve_backend, to_numpy
from tp4.cli import main
from tp4.errors import ConfigError


def _fake_cupy(devices: int = 1) -> types.ModuleType:
    """A stand-in `cupy` that delegates every array function to numpy."""
    fake = types.ModuleType("cupy")
    fake.__getattr__ = lambda name: getattr(np, name)  # PEP 562 module __getattr__
    fake.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: devices))
    return fake


@pytest.fixture
def no_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy -> ImportError


def _standardized(n=50, d=6, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(d, d))
    X = pd.DataFrame(rng.normal(size=(n, d)) @ mix, columns=[f"f{i}" for i in range(d)])
    return standardize(X)[0], list(X.columns)


def test_numpy_backend_is_the_default():
    backend = resolve_backend("numpy")
    assert isinstance(backend, Backend)
    assert backend.name == "numpy" and backend.requested == "numpy"
    assert backend.xp is np
    assert backend.fallback_reason is None
    assert resolve_backend().name == "numpy"
    assert backend.as_record() == {"requested": "numpy", "resolved": "numpy", "fallback_reason": None}


def test_cupy_missing_falls_back_to_numpy_with_warning_and_reason(no_cupy):
    with pytest.warns(RuntimeWarning, match="cupy"):
        backend = resolve_backend("cupy")
    assert backend.requested == "cupy"
    assert backend.name == "numpy"
    assert backend.xp is np
    assert "cupy" in backend.fallback_reason
    assert len(backend.fallback_reason) <= 200
    record = backend.as_record()
    assert record["requested"] == "cupy" and record["resolved"] == "numpy" and record["fallback_reason"]


def test_fake_cupy_with_a_device_is_selected(monkeypatch):
    fake = _fake_cupy(devices=1)
    monkeypatch.setitem(sys.modules, "cupy", fake)
    backend = resolve_backend("cupy")
    assert backend.name == "cupy" and backend.requested == "cupy"
    assert backend.xp is fake
    assert backend.fallback_reason is None


def test_fake_cupy_without_a_device_falls_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(devices=0))
    with pytest.warns(RuntimeWarning, match="no CUDA device"):
        backend = resolve_backend("cupy")
    assert backend.name == "numpy" and backend.xp is np
    assert "no CUDA device" in backend.fallback_reason


def test_cupy_runtime_error_falls_back(monkeypatch):
    fake = _fake_cupy()

    def boom():
        raise RuntimeError("driver version is insufficient " + "x" * 400)

    fake.cuda.runtime.getDeviceCount = boom
    monkeypatch.setitem(sys.modules, "cupy", fake)
    with pytest.warns(RuntimeWarning):
        backend = resolve_backend("cupy")
    assert backend.name == "numpy"
    assert backend.fallback_reason.startswith("cupy unavailable: RuntimeError")
    assert len(backend.fallback_reason) <= 200


def test_unknown_backend_raises_config_error():
    with pytest.raises(ConfigError, match="torch"):
        resolve_backend("torch")


def test_to_numpy_handles_arrays_and_get_objects():
    a = np.arange(3.0)
    assert to_numpy(a) is a

    class Device:
        def get(self):
            return np.array([1.0, 2.0])

    assert to_numpy(Device()).tolist() == [1.0, 2.0]
    assert to_numpy([1, 2]).tolist() == [1, 2]


@pytest.mark.parametrize("which", ["numpy", "fake_cupy"])
def test_eigen_crosscheck_matches_sklearn(which):
    Z, features = _standardized()
    result = fit_pca(Z, features)
    xp = np if which == "numpy" else _fake_cupy()
    assert eigen_crosscheck(Z, result.explained_variance_ratio, xp) < 1e-10


def test_eigen_crosscheck_uses_only_the_leading_ratios():
    Z, features = _standardized()
    result = fit_pca(Z, features, n_components=3)
    assert eigen_crosscheck(Z, result.explained_variance_ratio, np) < 1e-10


def test_cli_cupy_unavailable_runs_on_numpy_and_records_why(tmp_path, write_config, no_cupy):
    cfg = write_config("configs/pca.json", backend="cupy", plots=False)
    out = tmp_path / "runs"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    run = json.loads((out / "pca" / "run.json").read_text(encoding="utf-8"))
    assert run["backend"]["requested"] == "cupy"
    assert run["backend"]["resolved"] == "numpy"
    assert run["backend"]["fallback_reason"]
    summary = json.loads((out / "pca" / "summary.json").read_text(encoding="utf-8"))
    assert summary["backend_resolved"] == "numpy"
    assert summary["eigen_crosscheck_max_abs_diff"] < 1e-10


def test_cli_pca_default_backend_records_numpy(tmp_path, write_config):
    cfg = write_config("configs/pca.json", plots=False)
    out = tmp_path / "runs"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 0
    run = json.loads((out / "pca" / "run.json").read_text(encoding="utf-8"))
    assert run["backend"] == {"requested": "numpy", "resolved": "numpy", "fallback_reason": None}
    summary = json.loads((out / "pca" / "summary.json").read_text(encoding="utf-8"))
    assert summary["eigen_crosscheck_max_abs_diff"] < 1e-10


def test_cli_unknown_backend_still_exits_2(tmp_path, capsys, write_config):
    cfg = write_config("configs/pca.json", backend="torch")
    assert main(["run", str(cfg), "--out-root", str(tmp_path / "runs")]) == 2
    assert "backend" in capsys.readouterr().err


def test_real_cupy_when_importable():
    pytest.importorskip("cupy")
    backend = resolve_backend("cupy")
    if backend.name == "numpy":
        assert backend.fallback_reason
        return
    Z, features = _standardized()
    result = fit_pca(Z, features)
    assert eigen_crosscheck(Z, result.explained_variance_ratio, backend.xp) < 1e-8
