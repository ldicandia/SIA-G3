"""End-to-end: the tanh perceptron fits y = tanh(x) (VAL-03), asserted from the run JSON (VAL-09).

Every test runs the binary with its pinned defaults only (no learning-rate or
epoch overrides), so a passing test means the shipped defaults converge.
"""

from __future__ import annotations

# Strict upper bound on the final MSE (K-13); an MSE exactly equal to it fails.
TANH_MSE_MAX = 1e-2


def test_tanh_reaches_low_mse(run_validate) -> None:
    data = run_validate("tanh").data
    assert data["loss_per_epoch"][-1] < TANH_MSE_MAX


def test_tanh_json_shape(run_validate) -> None:
    data = run_validate("tanh").data
    hp = data["hyperparameters"]
    predictions = data["predictions"]

    assert data["case"] == "tanh"
    assert hp["activation"] == "tanh"
    assert hp["n_inputs"] == 1
    assert len(predictions) == 50
    assert all(len(p["input"]) == 1 for p in predictions)
    assert all(-2 <= p["input"][0] <= 2 for p in predictions)
    assert all(-1 < p["expected"] < 1 for p in predictions)
    assert len(data["loss_per_epoch"]) == hp["epochs"]


def test_tanh_loss_decreases_overall(run_validate) -> None:
    loss = run_validate("tanh").data["loss_per_epoch"]
    assert loss[-1] < loss[0]
