"""Plot the loss-per-epoch curve (and, for one-input runs, the fit) of every run JSON in a directory.

usage: python scripts/plot_validation.py [--in runs_validation] [--out plots]

Reads `<in>/*.json` written by `tp3 validate` and writes `<out>/<stem>_loss.png`
for each. When every prediction of a run has a single input (linear, tanh) it
also writes `<out>/<stem>_fit.png`: the samples as a scatter of (x, expected)
and the perceptron's own predictions as a line. Everything drawn comes from the
JSON — nothing is recomputed in Python. Only this script imports matplotlib;
the C++ core knows nothing of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)


def plot_loss(data: dict, out_path: Path) -> None:
    """Write the loss curve of one run (epoch index 1..N in array order)."""
    loss = data["loss_per_epoch"]
    hp = data["hyperparameters"]
    epochs = range(1, len(loss) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, loss, marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"{data['case']} — {hp['activation']}, lr={hp['learning_rate']}, seed={data['seed']}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def has_scalar_inputs(data: dict) -> bool:
    """True when every prediction's input has exactly one element (a plottable x)."""
    predictions = data["predictions"]
    return bool(predictions) and all(len(p["input"]) == 1 for p in predictions)


def plot_fit(data: dict, out_path: Path) -> None:
    """Write samples (x, expected) as a scatter and the run's predictions as a line sorted by x.

    Only `predictions` is read: the curve is what the C++ engine predicted, not a
    Python re-evaluation of the weights.
    """
    points = sorted((p["input"][0], p["expected"], p["predicted"]) for p in data["predictions"])
    xs = [x for x, _, _ in points]
    expected = [e for _, e, _ in points]
    predicted = [o for _, _, o in points]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, expected, s=18, label="samples", zorder=3)
    ax.plot(xs, predicted, color="C1", linewidth=1.5, label="perceptron", zorder=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{data['case']} — fit vs samples (seed={data['seed']})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_xor_comparison(in_dir: Path, out_path: Path) -> bool:
    """Plot [2,2,1], [2,3,2,1] and step loss curves on the same axes if present."""
    xor_specs = [
        ("xor_221.json", "MLP [2, 2, 1] (tanh)"),
        ("xor_2321.json", "MLP [2, 3, 2, 1] (tanh)"),
        ("xor_step.json", "Simple Perceptron (step)"),
    ]
    runs = []
    for filename, label in xor_specs:
        path = in_dir / filename
        if path.is_file():
            data = json.loads(path.read_text())
            runs.append((label, data))

    if len(runs) < 2:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    seed = runs[0][1].get("seed", 42)
    for label, data in runs:
        loss = data["loss_per_epoch"]
        epochs = range(1, len(loss) + 1)
        ax.plot(epochs, loss, label=label, linewidth=1.5)

    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"XOR — Loss Comparison (seed={seed})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="in_dir", default="runs_validation", help="directory holding run JSON files")
    parser.add_argument("--out", dest="out_dir", default="plots", help="directory that receives the PNGs")
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    json_paths = sorted(in_dir.glob("*.json")) if in_dir.is_dir() else []
    if not json_paths:
        print(f"no run JSON found in {in_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    has_xor_runs = plot_xor_comparison(in_dir, out_dir / "xor_loss.png")
    if has_xor_runs:
        print(f"wrote: {out_dir / 'xor_loss.png'}")

    for path in json_paths:
        # If combined xor_loss.png was already drawn, don't overwrite it with a single xor.json plot
        if has_xor_runs and path.stem == "xor":
            continue
        data = json.loads(path.read_text())
        png = out_dir / f"{path.stem}_loss.png"
        plot_loss(data, png)
        print(f"wrote: {png}")
        if has_scalar_inputs(data):
            fit_png = out_dir / f"{path.stem}_fit.png"
            plot_fit(data, fit_png)
            print(f"wrote: {fit_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
