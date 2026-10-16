"""Experiment kind `hopfield_qr`: the five menu QR codes in a Hopfield memory (Ej. 2.1).

Assignment items (docs/SIATP42026.pdf pages 4-5, section 2.1): store the five restaurant
menu QR codes (food, wine, beverages, dessert, specials), build query patterns, add random
noise at different levels, and analyze the iterations and the energy at each step.

Conventions:
- Each PNG is a 37 x 37 code read with `tp4.analysis.qr.load_qr_image`: black = +1,
  white = -1, N = 1369 neurons, no resizing and no quiet zone.
- Updates are asynchronous by default (one random permutation of the neurons per sweep).
- `iterations` counts the sweeps that changed the state; the last, unchanged sweep is the
  convergence check.

Learning rules (params.rules):
- `hebb`: the course's Hebbian rule, W = (1/N) sum xi xi^T with zero diagonal.
- `pseudoinverse`: the projection rule, a comparison rule for correlated patterns
  (Personnaz et al., 1985). It is the same network with another learning rule. Drop it
  with `"rules": ["hebb"]`.

Config keys (besides common keys):
  images   list of 2 to 10 distinct .png paths, in the order the codes are stored

Artifacts:
  stored.csv, queries.csv, energy.csv, recovery_rates.csv, patterns.png,
  recovery_vs_noise.png, summary.json, run.json
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.hopfield_eval import (
    classify_states,
    flip_bits,
    n_flips_for,
    nearest_pattern,
    overlaps,
    stability_margins,
)
from ..analysis.qr import load_qr_image, matrix_to_image, qr_name
from ..errors import ConfigError, DataError
from ..models.hopfield import (
    MAX_ITER,
    RULES,
    UPDATES,
    HopfieldConfig,
    energy,
    is_fixed_point,
    recall,
    sign_keep,
    train,
)
from ..parallel import spawn_seeds
from ..plotting import plt, save_figure
from ..runs import Experiment, RunContext, resolve_project_path, write_csv

EXTRA_KEYS = frozenset({"images"})
PARAM_DEFAULTS: dict[str, Any] = {
    "side": 37,
    "rules": ["hebb", "pseudoinverse"],
    "update": "async",
    "max_iter": 100,
    "noise_levels": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    "trials": 20,
}
MAX_IMAGES = 10
MAX_RECALLS = 50000
MAX_SIDE = 64
MAX_LEVELS = 21
MAX_TRIALS = 200
ENERGY_TOL = 1e-9

RULE_COLORS = {"hebb": "#D55E00", "pseudoinverse": "#0072B2"}
RULE_LABELS = {"hebb": "Hebb (regla del curso)", "pseudoinverse": "Pseudoinversa (comparación)"}


def _is_int(v: Any) -> bool:
    return type(v) is int


def _is_real(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def _err(msg: str) -> ConfigError:
    return ConfigError(f"hopfield_qr: {msg}")


def validate(config: dict) -> None:
    images = config.get("images")
    if not isinstance(images, list):
        raise _err(f"images must be a list of .png paths, got {images!r}")
    if not 2 <= len(images) <= MAX_IMAGES:
        raise _err(f"images must hold between 2 and {MAX_IMAGES} paths, got {len(images)}")
    for item in images:
        if not isinstance(item, str) or not item or not item.lower().endswith(".png"):
            raise _err(f"every image must be a non-empty path ending in .png, got {item!r}")
    if len(set(images)) != len(images):
        raise _err(f"images must be distinct, got {images!r}")
    names = [qr_name(item) for item in images]
    if len(set(names)) != len(names):
        raise _err(f"image names must be distinct, got {names!r}")
    for item in images:
        if not resolve_project_path(item).is_file():
            raise _err(f"image not found: {item}")

    p = config["params"]
    side = p["side"]
    if not (_is_int(side) and 5 <= side <= MAX_SIDE):
        raise _err(f"params.side must be an integer in [5, {MAX_SIDE}], got {side!r}")

    rules = p["rules"]
    if not isinstance(rules, list) or not rules:
        raise _err("params.rules must be a non-empty list")
    if len(set(map(str, rules))) != len(rules) or any(r not in RULES for r in rules):
        raise _err(f"params.rules must contain distinct values from {list(RULES)}, got {rules!r}")

    if p["update"] not in UPDATES:
        raise _err(f"params.update must be one of {list(UPDATES)}, got {p['update']!r}")

    max_iter = p["max_iter"]
    if not (_is_int(max_iter) and 1 <= max_iter <= MAX_ITER):
        raise _err(f"params.max_iter must be an integer in [1, {MAX_ITER}], got {max_iter!r}")

    levels = p["noise_levels"]
    if not isinstance(levels, list) or not levels:
        raise _err("params.noise_levels must be a non-empty list")
    if len(levels) > MAX_LEVELS:
        raise _err(f"params.noise_levels may hold at most {MAX_LEVELS} values, got {len(levels)}")
    if not all(_is_real(lv) and 0.0 <= float(lv) <= 1.0 for lv in levels):
        raise _err(f"params.noise_levels must contain numbers in [0, 1], got {levels!r}")
    if any(not float(a) < float(b) for a, b in zip(levels, levels[1:], strict=False)):
        raise _err(f"params.noise_levels must be strictly increasing, got {levels!r}")

    trials = p["trials"]
    if not (_is_int(trials) and 1 <= trials <= MAX_TRIALS):
        raise _err(f"params.trials must be an integer in [1, {MAX_TRIALS}], got {trials!r}")

    n_recalls = len(images) * (len(levels) * trials) * len(rules)
    if n_recalls > MAX_RECALLS:
        raise _err(f"the grid needs {n_recalls} recalls, above MAX_RECALLS ({MAX_RECALLS})")


# --------------------------------------------------------------------------- queries


def _flip_cells(
    P: np.ndarray, levels: list[float], trials: int, seeds: list[int]
) -> list[dict[str, Any]]:
    """One cell per (level, pattern), level-major. Each cell owns its seed and its queries."""
    n_patterns, n_neurons = P.shape
    cells: list[dict[str, Any]] = []
    cursor = 0
    for level in levels:
        k = n_flips_for(level, n_neurons)
        for pattern_idx in range(n_patterns):
            seed = seeds[cursor]
            cursor += 1
            queries = flip_bits(P[pattern_idx], k, trials, np.random.default_rng(seed))
            cells.append(
                {
                    "damage": "flip",
                    "level": float(level),
                    "size": None,
                    "seed": seed,
                    "pattern": [pattern_idx] * trials,
                    "corner": [None] * trials,
                    "top": [None] * trials,
                    "left": [None] * trials,
                    "trial": list(range(trials)),
                    "queries": queries,
                }
            )
    return cells


# --------------------------------------------------------------------------- plots


def _plot_patterns(images: list[Any], P: np.ndarray, side: int, path: Path) -> None:
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    axes = np.asarray(axes).reshape(2, n)
    for col, img in enumerate(images):
        black = float((P[col] == 1).mean())
        axes[0, col].imshow(img.pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0, col].set_title(f"{img.name}\nimagen original", fontsize=9)
        axes[1, col].imshow(P[col].reshape(side, side), cmap="binary", vmin=-1, vmax=1, interpolation="nearest")
        axes[1, col].set_title(f"matriz ±1 (negro = +1)\nnegro {black:.3f}", fontsize=9)
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(f"Los {n} códigos QR: imagen y matriz {side}×{side} que se almacena (N = {side * side})", fontsize=11)
    fig.tight_layout(h_pad=2.5)
    save_figure(fig, path)


def _plot_recovery(rates: list[dict[str, Any]], rules: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for rule in rules:
        sub = [r for r in rates if r["damage"] == "flip" and r["rule"] == rule]
        x = [r["level"] for r in sub]
        ax.plot(
            x,
            [r["recovery_rate"] for r in sub],
            marker="o",
            linewidth=2.0,
            color=RULE_COLORS[rule],
            label=f"Recuperación exacta: {RULE_LABELS[rule]}",
        )
        if rule == "hebb":
            ax.plot(
                x,
                [r["mean_attractor_rate"] for r in sub],
                linestyle="--",
                marker="x",
                linewidth=1.6,
                color="#444444",
                label="Hebb: termina en el «QR promedio» sign(Σ ξ)",
            )
    ax.set_xlabel("Nivel de ruido (fracción de módulos invertidos)")
    ax.set_ylabel("Tasa de recuperación exacta")
    ax.set_ylim(-0.04, 1.06)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", fontsize=8.5)
    ax.set_title("Recuperación de los códigos QR frente al ruido (actualización asincrónica)")
    fig.tight_layout()
    save_figure(fig, path)


# --------------------------------------------------------------------------- run


def run(ctx: RunContext) -> dict[str, Any]:
    p = ctx.params
    side = int(p["side"])
    rules = [str(r) for r in p["rules"]]
    update = str(p["update"])
    max_iter = int(p["max_iter"])
    levels = [float(x) for x in p["noise_levels"]]
    trials = int(p["trials"])

    images = [load_qr_image(path, side) for path in ctx.config["images"]]
    names = [img.name for img in images]
    P = np.stack([img.matrix.reshape(-1) for img in images]).astype(np.int8)
    n_patterns, n_neurons = P.shape
    roundtrip_identical = all(
        bool(np.array_equal(matrix_to_image(img.matrix), img.pixels)) for img in images
    )
    if len({row.tobytes() for row in P}) != n_patterns:
        raise DataError("hopfield_qr: two of the images hold the same code")

    mean_attractor = np.where(P.sum(axis=0) >= 0, 1, -1).astype(np.int8)

    weights: dict[str, np.ndarray] = {}
    stored_rows: list[dict[str, Any]] = []
    rule_stats: dict[str, dict[str, Any]] = {}
    for rule in rules:
        try:
            W = train(P, rule)
        except ValueError as exc:
            raise DataError(f"hopfield_qr: cannot store the codes with rule {rule}: {exc}") from exc
        weights[rule] = W
        fixed = is_fixed_point(W, P)
        margins = stability_margins(W, P)
        unstable = np.sum(sign_keep(P.astype(np.float64) @ W, P) != P, axis=1)
        energies = np.asarray(energy(W, P), dtype=float)
        rule_stats[rule] = {"n_fixed": int(np.sum(fixed)), "unstable": [int(u) for u in unstable]}
        for k, name in enumerate(names):
            stored_rows.append(
                {
                    "rule": rule,
                    "pattern": name,
                    "fixed_point": bool(fixed[k]),
                    "n_unstable_bits": int(unstable[k]),
                    "stability_margin": round(float(margins[k]), 4),
                    "energy": round(float(energies[k]), 4),
                }
            )
    write_csv(ctx.artifact("stored.csv"), pd.DataFrame(stored_rows))

    # ctx.rng is not used. One spawn_seeds list is consumed in cell order (see _flip_cells).
    n_cells = len(levels) * n_patterns
    seeds = spawn_seeds(ctx.seed, n_cells)
    cells = _flip_cells(P, levels, trials, seeds)

    query_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, Any, str], list[dict[str, Any]]] = {}
    energy_never_increased = True
    query_id = 0
    for cell in cells:
        queries = cell["queries"]
        m = queries.shape[0]
        per_rule: dict[str, dict[str, Any]] = {}
        for r_idx, rule in enumerate(rules):
            res = recall(
                weights[rule],
                queries,
                HopfieldConfig(rule=rule, update=update, max_iter=max_iter),
                np.random.default_rng([cell["seed"], r_idx]),
            )
            classes, class_idx = classify_states(res.states, P)
            per_rule[rule] = {
                "res": res,
                "classes": classes,
                "class_idx": class_idx,
                "ov": overlaps(res.states, P),
                "near": nearest_pattern(res.states, P),
            }
        for i in range(m):
            target = cell["pattern"][i]
            n_changed = int(np.sum(queries[i] != P[target]))
            for rule in rules:
                info = per_rule[rule]
                res = info["res"]
                final = res.states[i]
                cls = str(info["classes"][i])
                cidx = int(info["class_idx"][i])
                e_seq = res.energies[i]
                if update == "async" and e_seq.size > 1 and bool(np.any(np.diff(e_seq) > ENERGY_TOL)):
                    energy_never_increased = False
                recovered = bool(np.array_equal(final, P[target]))
                row = {
                    "query_id": query_id,
                    "damage": cell["damage"],
                    "level": cell["level"],
                    "size": cell["size"],
                    "corner": cell["corner"][i],
                    "top": cell["top"][i],
                    "left": cell["left"][i],
                    "pattern": names[target],
                    "trial": cell["trial"][i],
                    "n_changed": n_changed,
                    "damaged_frac": round(n_changed / n_neurons, 4),
                    "rule": rule,
                    "iterations": int(res.iterations[i]),
                    "status": str(res.status[i]),
                    "final_class": cls,
                    "final_pattern": names[cidx] if cidx >= 0 else "",
                    "recovered": recovered,
                    "overlap_target": round(float(info["ov"][i, target]), 4),
                    "hamming_target": int(np.sum(final != P[target])),
                    "nearest_pattern": names[int(info["near"][i])],
                    "is_mean_attractor": bool(np.array_equal(final, mean_attractor)),
                    "final_energy": float(e_seq[-1]),
                }
                query_rows.append(row)
                key = (cell["damage"], cell["level"] if cell["size"] is None else cell["size"], rule)
                groups.setdefault(key, []).append(row)
                for it_num, e_val in enumerate(e_seq):
                    energy_rows.append(
                        {"query_id": query_id, "rule": rule, "iteration": int(it_num), "energy": float(e_val)}
                    )
            query_id += 1

    queries_df = pd.DataFrame(query_rows)
    for col in ("size", "top", "left"):
        queries_df[col] = queries_df[col].astype("Int64")
    write_csv(ctx.artifact("queries.csv"), queries_df)
    write_csv(ctx.artifact("energy.csv"), pd.DataFrame(energy_rows))

    rate_rows: list[dict[str, Any]] = []
    for (damage, key, rule), rows in groups.items():
        count = len(rows)

        def frac(pred, rows=rows, count=count) -> float:
            return round(sum(1 for r in rows if pred(r)) / count, 4)

        iters = [r["iterations"] for r in rows]
        rate_rows.append(
            {
                "damage": damage,
                "level": key if damage == "flip" else None,
                "size": None if damage == "flip" else key,
                "rule": rule,
                "n_queries": count,
                "mean_damaged_frac": round(float(np.mean([r["damaged_frac"] for r in rows])), 4),
                "recovery_rate": frac(lambda r: r["recovered"]),
                "other_rate": frac(lambda r: r["final_class"] == "stored" and not r["recovered"]),
                "inverted_rate": frac(lambda r: r["final_class"] == "inverted"),
                "spurious_rate": frac(lambda r: r["final_class"] == "spurious"),
                "mean_attractor_rate": frac(lambda r: r["is_mean_attractor"]),
                "max_iter_rate": frac(lambda r: r["status"] == "max_iter"),
                "mean_iterations": round(float(np.mean(iters)), 4),
                "max_iterations": int(max(iters)),
                "mean_overlap_target": round(float(np.mean([r["overlap_target"] for r in rows])), 4),
            }
        )
    rates_df = pd.DataFrame(rate_rows)
    rates_df["size"] = rates_df["size"].astype("Int64")
    write_csv(ctx.artifact("recovery_rates.csv"), rates_df)

    def flip_curve(rule: str) -> list[float]:
        return [r["recovery_rate"] for r in rate_rows if r["damage"] == "flip" and r["rule"] == rule]

    def max_full_level(rule: str) -> float | None:
        best: float | None = None
        for lv, rate in zip(levels, flip_curve(rule), strict=True):
            if rate != 1.0:
                break
            best = lv
        return best

    if ctx.plots:
        _plot_patterns(images, P, side, ctx.artifact("patterns.png"))
        _plot_recovery(rate_rows, rules, ctx.artifact("recovery_vs_noise.png"))

    summary: dict[str, Any] = {
        "n_patterns": int(n_patterns),
        "n_neurons": int(n_neurons),
        "side": side,
        "pattern_names": names,
        "roundtrip_identical": bool(roundtrip_identical),
        "black_fractions": [round(float((P[k] == 1).mean()), 4) for k in range(n_patterns)],
        "capacity_0138": round(0.138 * n_neurons, 4),
        "capacity_015": round(0.15 * n_neurons, 4),
        "rules": rules,
        "update": update,
        "noise_levels": levels,
        "trials": trials,
        "n_queries": int(query_id),
        "n_recalls": int(query_id * len(rules)),
    }
    for rule in rules:
        rule_rows = [r for r in query_rows if r["rule"] == rule]
        summary[f"{rule}_n_fixed_points"] = rule_stats[rule]["n_fixed"]
        summary[f"{rule}_unstable_bits"] = rule_stats[rule]["unstable"]
        summary[f"{rule}_flip_recovery"] = flip_curve(rule)
        summary[f"{rule}_mean_iterations"] = round(float(np.mean([r["iterations"] for r in rule_rows])), 4)
        summary[f"{rule}_max_level_full_recovery"] = max_full_level(rule)
    hebb_rows = [r for r in query_rows if r["rule"] == "hebb"]
    summary["hebb_frac_mean_attractor"] = (
        round(sum(1 for r in hebb_rows if r["is_mean_attractor"]) / len(hebb_rows), 4) if hebb_rows else None
    )
    summary["max_iterations"] = int(max(r["iterations"] for r in query_rows))
    summary["energy_never_increased"] = bool(energy_never_increased) if update == "async" else None
    summary["backend_resolved"] = ctx.backend.name
    return summary


EXPERIMENT = Experiment(
    kind="hopfield_qr",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
