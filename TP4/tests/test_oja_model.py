"""Hand-written Oja neuron: recovers the leading eigenvector, keeps its norm near 1, flags divergence."""

from __future__ import annotations

import ast
import re
import warnings

import numpy as np
import pytest

from tp4.models import oja as oja_module
from tp4.models.oja import MAX_W_NORM, OjaConfig, convergence_epoch, train_oja
from tp4.runs import PROJECT_ROOT

EIGENVALUES = np.array([5.0, 2.0, 1.0, 0.5, 0.25])


def _anisotropic(n: int = 500, seed: int = 0):
    """Centered Gaussian with known covariance spectrum in a random orthonormal basis."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(EIGENVALUES.size, EIGENVALUES.size)))
    X = rng.normal(size=(n, EIGENVALUES.size)) * np.sqrt(EIGENVALUES) @ basis.T
    return X - X.mean(axis=0)


def _leading_eigenvector(X):
    cov = X.T @ X / X.shape[0]
    values, vectors = np.linalg.eigh(cov)
    return vectors[:, int(np.argmax(values))]


def _cfg(**kw):
    base = dict(epochs=100, eta0=1e-3, eta_min=1e-5, eta_decay="constant")
    base.update(kw)
    return OjaConfig(**base)


# --------------------------------------------------------------------------- recovery (D-07)


def test_recovers_the_leading_eigenvector_and_norm_one():
    X = _anisotropic()
    result = train_oja(X, _cfg(), np.random.default_rng(42))
    v1 = _leading_eigenvector(X)
    w = result.weights
    cos = abs(w @ v1) / np.linalg.norm(w)
    assert cos >= 0.99
    assert abs(np.linalg.norm(w) - 1.0) <= 0.01
    assert result.diverged is False and result.divergence_epoch is None


def test_history_has_one_entry_per_epoch_and_epochs_run_from_one():
    X = _anisotropic()
    reference = _leading_eigenvector(X)
    result = train_oja(X, _cfg(epochs=17), np.random.default_rng(1), reference=reference)
    assert set(result.history) == {"epoch", "eta", "w_norm", "delta_w", "cos_to_reference"}
    for column in result.history.values():
        assert column.shape == (17,)
    assert result.history["epoch"].tolist() == list(range(1, 18))
    assert np.isfinite(result.history["cos_to_reference"]).all()
    assert (result.history["cos_to_reference"] <= 1.0 + 1e-12).all()


def test_without_reference_the_cosine_column_is_nan():
    result = train_oja(_anisotropic(), _cfg(epochs=3), np.random.default_rng(1))
    assert np.isnan(result.history["cos_to_reference"]).all()


def test_initial_weights_are_uniform_01_and_seeded():
    X = _anisotropic()
    a = train_oja(X, _cfg(epochs=2), np.random.default_rng(9))
    b = train_oja(X, _cfg(epochs=2), np.random.default_rng(9))
    assert (a.initial_weights >= 0).all() and (a.initial_weights < 1).all()
    assert np.array_equal(a.initial_weights, np.random.default_rng(9).uniform(0.0, 1.0, size=X.shape[1]))
    assert np.array_equal(a.initial_weights, b.initial_weights)


def test_same_seed_gives_identical_weights_and_history():
    X = _anisotropic()
    a = train_oja(X, _cfg(), np.random.default_rng(5))
    b = train_oja(X, _cfg(), np.random.default_rng(5))
    assert np.array_equal(a.weights, b.weights)
    for key in a.history:
        assert np.array_equal(a.history[key], b.history[key], equal_nan=True)
    c = train_oja(X, _cfg(), np.random.default_rng(6))
    assert not np.array_equal(a.initial_weights, c.initial_weights)


def test_the_online_rule_is_applied_one_sample_at_a_time():
    """One epoch on 2 samples must equal two manual Oja steps in the rng's permutation order."""
    X = np.array([[1.0, 0.5], [-1.0, -0.5], [0.3, -0.2]])
    cfg = _cfg(epochs=1, eta0=0.01)
    result = train_oja(X, cfg, np.random.default_rng(3))
    rng = np.random.default_rng(3)
    w = rng.uniform(0.0, 1.0, size=2)
    for i in rng.permutation(3):
        y = w @ X[i]
        w = w + 0.01 * y * (X[i] - y * w)
    assert np.allclose(result.weights, w, atol=1e-15, rtol=0)


# --------------------------------------------------------------------------- schedule


def test_exponential_decay_runs_from_eta0_to_eta_min():
    X = _anisotropic(n=60)
    result = train_oja(X, _cfg(epochs=20, eta0=5e-3, eta_min=1e-4, eta_decay="exponential"), np.random.default_rng(0))
    eta = result.history["eta"]
    assert eta[-1] == pytest.approx(1e-4, rel=1e-12)
    assert eta[0] < 5e-3 and eta[0] > eta[-1]
    assert (np.diff(eta) < 0).all()


def test_constant_eta_is_flat():
    result = train_oja(_anisotropic(n=60), _cfg(epochs=5), np.random.default_rng(0))
    assert (result.history["eta"] == 1e-3).all()


# --------------------------------------------------------------------------- divergence


def test_divergence_is_flagged_not_raised_and_no_warning_escapes():
    X = 10.0 * np.random.default_rng(2).normal(size=(200, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = train_oja(X, _cfg(epochs=50, eta0=0.5), np.random.default_rng(0))
    assert result.diverged is True
    assert isinstance(result.divergence_epoch, int) and 1 <= result.divergence_epoch <= 50
    assert np.isnan(result.weights).all()
    h = result.history
    assert h["epoch"].size == result.divergence_epoch
    assert np.isnan(h["w_norm"][-1]) and np.isnan(h["delta_w"][-1]) and np.isnan(h["cos_to_reference"][-1])
    assert MAX_W_NORM == 1e6


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "cfg",
    [
        dict(epochs=0),
        dict(epochs=True),
        dict(epochs=2.5),
        dict(eta0=0.0),
        dict(eta0=float("nan")),
        dict(eta_min=0.0),
        dict(eta_min=2e-3),
        dict(eta_decay="cosine"),
    ],
)
def test_bad_config_raises_value_error(cfg):
    with pytest.raises(ValueError):
        train_oja(_anisotropic(n=20), _cfg(**cfg), np.random.default_rng(0))


@pytest.mark.parametrize(
    "X",
    [
        np.arange(6, dtype=float),
        np.ones((1, 3)),
        np.ones((5, 0)),
        np.array([[1.0, np.nan], [0.0, 1.0], [2.0, 1.0]]),
        np.array([[1.0, np.inf], [0.0, 1.0], [2.0, 1.0]]),
    ],
)
def test_bad_data_raises_value_error(X):
    with pytest.raises(ValueError):
        train_oja(X, _cfg(), np.random.default_rng(0))


def test_reference_must_match_the_feature_count_and_be_usable():
    X = _anisotropic(n=20)
    for reference in (np.ones(4), np.zeros(5), np.full(5, np.nan)):
        with pytest.raises(ValueError):
            train_oja(X, _cfg(), np.random.default_rng(0), reference=reference)


# --------------------------------------------------------------------------- convergence_epoch


def test_convergence_epoch_is_first_epoch_after_which_all_stay_inside():
    assert convergence_epoch([3.0, 1.2, 1.005, 0.999, 1.0], 0.01) == 3
    assert convergence_epoch([1.0, 1.0, 1.5], 0.01) is None
    assert convergence_epoch([1.0, 1.5, 1.0, 1.0], 0.01) == 3


def test_convergence_band_is_inclusive():
    assert convergence_epoch([2.0, 1.25, 0.75], 0.25) == 2


def test_convergence_epoch_none_for_empty_or_nan_tail():
    assert convergence_epoch([], 0.01) is None
    assert convergence_epoch([1.0, 1.0, float("nan")], 0.01) is None
    assert convergence_epoch([float("nan"), 1.0, 1.0], 0.01) == 2


# --------------------------------------------------------------------------- hand-written (OJA-01)

ALLOWED_TOP_LEVEL = {"__future__", "math", "dataclasses", "typing", "numpy", "tp4"}
EIGEN_CALL = re.compile(r"linalg\s*\.\s*(eig|eigh|eigvals|eigvalsh|svd|svdvals)\b|\b(eigh?|svd)\s*\(")


def test_oja_module_imports_only_numpy_and_stdlib():
    tree = ast.parse((PROJECT_ROOT / "tp4" / "models" / "oja.py").read_text(encoding="utf-8"))
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            tops.add((node.module or "").split(".")[0])
    assert tops <= ALLOWED_TOP_LEVEL, tops - ALLOWED_TOP_LEVEL


def test_oja_module_never_calls_an_eigen_or_svd_routine():
    text = (PROJECT_ROOT / "tp4" / "models" / "oja.py").read_text(encoding="utf-8")
    assert not EIGEN_CALL.search(text), EIGEN_CALL.search(text)
    assert "sklearn" not in text and "scipy" not in text
    assert oja_module.MAX_W_NORM == 1e6
