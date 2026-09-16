"""End-to-end: the identity perceptron fits y = x (VAL-02), asserted from the run JSON (VAL-09).

Every test runs the binary with its pinned defaults only (no learning-rate or
epoch overrides), so a passing test means the shipped defaults converge.
"""

from __future__ import annotations

# Strict upper bound on the final MSE (K-13); an MSE exactly equal to it fails.
LINEAR_MSE_MAX = 1e-3


def test_linear_reaches_low_mse(run_validate) -> None:
    data = run_validate("linear").data
    assert data["loss_per_epoch"][-1] < LINEAR_MSE_MAX


def test_linear_json_shape(run_validate) -> None:
    data = run_validate("linear").data
    hp = data["hyperparameters"]
    predictions = data["predictions"]

    assert data["case"] == "linear"
    assert hp["activation"] == "identity"
    assert hp["n_inputs"] == 1
    assert len(predictions) == 50
    assert all(len(p["input"]) == 1 for p in predictions)
    assert all(-2 <= p["input"][0] <= 2 for p in predictions)
    assert len(data["loss_per_epoch"]) == hp["epochs"]
    assert len(data["final_weights"]) == 1


def test_linear_weights_approach_identity(run_validate) -> None:
    data = run_validate("linear").data
    assert abs(data["final_weights"][0] - 1.0) < 0.05
    assert abs(data["bias"]) < 0.05
