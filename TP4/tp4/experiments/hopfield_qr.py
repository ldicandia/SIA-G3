"""Experiment kind `hopfield_qr`: the five menu QR codes in a Hopfield memory (Ej. 2.1).

Assignment items (docs/SIATP42026.pdf pages 4-5, section 2.1): store the five restaurant
menu QR codes (food, wine, beverages, dessert, specials), build query patterns, add random
noise at different levels, and analyze the iterations and the energy at each step. The
story: one corner of the table's QR is torn, the others are worn, scratched or stained.

Conventions:
- Each PNG is a 37 x 37 code read with `tp4.analysis.qr.load_qr_image`: black = +1,
  white = -1, N = 1369 neurons, no resizing and no quiet zone.
- Updates are asynchronous by default (one random permutation of the neurons per sweep).
- `iterations` counts the sweeps that changed the state; the last, unchanged sweep is the
  convergence check and adds no energy.csv row.
- A query is recovered only when its final state equals the stored target exactly.

Learning rules (params.rules):
- `hebb`: the course's Hebbian rule, W = (1/N) sum xi xi^T with zero diagonal.
- `pseudoinverse`: the projection rule, a comparison rule for correlated patterns
  (Personnaz et al., 1985). It is the same network with another learning rule. Drop it
  with `"rules": ["hebb"]`.

Damaged queries (every rule sees exactly the same ones):
- `flip`: round(level * N) distinct modules flipped, `trials` per code and level (wear).
- `torn_corner`: the paper triangle row + col < size missing (white) at each corner.
- `occlusion`: a black size x size stain at a seeded random position, `trials` per code.

Config keys (besides common keys):
  images   list of 2 to 10 distinct .png paths, in the order the codes are stored

Artifacts:
  stored.csv, overlaps.csv, queries.csv, energy.csv, energy_trace.csv, recovery_rates.csv,
  patterns.png, overlaps.png, shared_modules.png, hebb_attractor.png, damage_examples.png,
  recovery_vs_noise.png, iterations.png, energy_per_step.png, summary.json, run.json
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.hopfield_eval import (
    classify_states,
    dot_matrix,
    flip_bits,
    n_flips_for,
    nearest_pattern,
    overlaps,
    stability_margins,
)
from ..analysis import qr as qrmod
from ..analysis.qr import (
    DEFAULT_QUIET_ZONE,
    DEFAULT_SCALE,
    MAX_QUIET_ZONE,
    MAX_SCALE,
    load_qr_image,
    matrix_to_image,
    qr_name,
)
from ..analysis.qr_damage import CORNERS, random_occlusions, torn_corner
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
    "corners": list(CORNERS),
    "torn_corner_sizes": [8, 12, 16, 20, 24, 28, 32],
    "occlusion_sizes": [8, 12, 16, 20, 24],
    "trace_levels": [0.1, 0.3, 0.45],
    "decode": True,
    "decode_scale": DEFAULT_SCALE,
    "decode_quiet_zone": DEFAULT_QUIET_ZONE,
}
MAX_IMAGES = 10
MAX_RECALLS = 50000
MAX_SIDE = 64
MAX_LEVELS = 21
MAX_TRIALS = 200
ENERGY_TOL = 1e-9
DAMAGES = ("flip", "torn_corner", "occlusion")

RULE_COLORS = {"hebb": "#D55E00", "pseudoinverse": "#0072B2"}
RULE_LABELS = {"hebb": "Hebb (regla del curso)", "pseudoinverse": "Pseudoinversa (comparación)"}
RULE_SHORT = {"hebb": "Hebb", "pseudoinverse": "Pseudoinversa"}
ATTRACTOR_COLOR = "#444444"
SHARED_COLOR = "#E69F00"
UNSTABLE_COLOR = "#D55E00"

TRACE_COLUMNS = ["rule", "level", "pattern", "step", "sweep", "energy"]


def _is_int(v: Any) -> bool:
    return type(v) is int


def _is_real(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def _err(msg: str) -> ConfigError:
    return ConfigError(f"hopfield_qr: {msg}")


def _same_level(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)


def _check_sizes(name: str, sizes: Any, side: int) -> None:
    if not isinstance(sizes, list):
        raise _err(f"params.{name} must be a list, got {sizes!r}")
    if not all(_is_int(s) and 1 <= s <= side for s in sizes):
        raise _err(f"params.{name} must hold integers in [1, {side}], got {sizes!r}")
    if any(not a < b for a, b in zip(sizes, sizes[1:], strict=False)):
        raise _err(f"params.{name} must be strictly increasing, got {sizes!r}")


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

    corners = p["corners"]
    if not isinstance(corners, list) or any(c not in CORNERS for c in corners):
        raise _err(f"params.corners must be a list of values from {list(CORNERS)}, got {corners!r}")
    if len(set(corners)) != len(corners):
        raise _err(f"params.corners must not repeat a corner, got {corners!r}")
    _check_sizes("torn_corner_sizes", p["torn_corner_sizes"], side)
    _check_sizes("occlusion_sizes", p["occlusion_sizes"], side)

    trace_levels = p["trace_levels"]
    if not isinstance(trace_levels, list):
        raise _err(f"params.trace_levels must be a list, got {trace_levels!r}")
    for lv in trace_levels:
        if not (_is_real(lv) and any(_same_level(lv, x) for x in levels)):
            raise _err(f"params.trace_levels must be taken from params.noise_levels {levels}, got {lv!r}")
    if len(set(float(x) for x in trace_levels)) != len(trace_levels):
        raise _err(f"params.trace_levels must not repeat a level, got {trace_levels!r}")
    if trace_levels and p["update"] != "async":
        raise _err("params.trace_levels needs params.update = 'async' (the per-update trace is asynchronous)")

    if type(p["decode"]) is not bool:
        raise _err(f"params.decode must be a boolean, got {p['decode']!r}")
    scale = p["decode_scale"]
    if not (_is_int(scale) and 1 <= scale <= MAX_SCALE):
        raise _err(f"params.decode_scale must be an integer in [1, {MAX_SCALE}], got {scale!r}")
    quiet = p["decode_quiet_zone"]
    if not (_is_int(quiet) and 0 <= quiet <= MAX_QUIET_ZONE):
        raise _err(f"params.decode_quiet_zone must be an integer in [0, {MAX_QUIET_ZONE}], got {quiet!r}")

    n_corner = len(corners) * len(p["torn_corner_sizes"])
    n_per_image = len(levels) * trials + n_corner + len(p["occlusion_sizes"]) * trials
    n_recalls = len(images) * n_per_image * len(rules)
    if n_recalls > MAX_RECALLS:
        raise _err(f"the grid needs {n_recalls} recalls, above MAX_RECALLS ({MAX_RECALLS})")


# --------------------------------------------------------------------------- queries


def _cell(damage: str, level: float | None, size: int | None, seed: int, queries: np.ndarray, **cols: list) -> dict:
    """One batch of damaged queries sharing a seed (noise draws and the per-rule async orders)."""
    m = queries.shape[0]
    return {
        "damage": damage,
        "level": level,
        "size": size,
        "seed": seed,
        "queries": queries,
        "pattern": cols["pattern"],
        "corner": cols.get("corner", [None] * m),
        "top": cols.get("top", [None] * m),
        "left": cols.get("left", [None] * m),
        "trial": cols["trial"],
    }


def _build_cells(
    P: np.ndarray,
    side: int,
    levels: list[float],
    trials: int,
    corners: list[str],
    corner_sizes: list[int],
    occlusion_sizes: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Cells in output order. Seeds are consumed in the same order: flip cells (level-major, one
    per code), then one cell per torn-corner size, then one per (stain size, code)."""
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
                _cell("flip", float(level), None, seed, queries, pattern=[pattern_idx] * trials, trial=list(range(trials)))
            )
    ordered_corners = [c for c in CORNERS if c in corners]
    if ordered_corners:
        for size in corner_sizes:
            seed = seeds[cursor]
            cursor += 1
            pattern_col: list[int] = []
            corner_col: list[str] = []
            rows = []
            for pattern_idx in range(n_patterns):
                for corner in ordered_corners:
                    rows.append(torn_corner(P[pattern_idx].reshape(side, side), size, corner).reshape(-1))
                    pattern_col.append(pattern_idx)
                    corner_col.append(corner)
            cells.append(
                _cell(
                    "torn_corner", None, int(size), seed, np.stack(rows),
                    pattern=pattern_col, corner=corner_col, trial=[0] * len(rows),
                )  # fmt: skip
            )
    for size in occlusion_sizes:
        for pattern_idx in range(n_patterns):
            seed = seeds[cursor]
            cursor += 1
            queries, positions = random_occlusions(
                P[pattern_idx].reshape(side, side), size, trials, np.random.default_rng(seed)
            )
            cells.append(
                _cell(
                    "occlusion", None, int(size), seed, queries,
                    pattern=[pattern_idx] * trials, top=[int(t) for t in positions[:, 0]],
                    left=[int(x) for x in positions[:, 1]], trial=list(range(trials)),
                )  # fmt: skip
            )
    return cells


def _count_cells(n_patterns: int, levels: list, corners: list, corner_sizes: list, occlusion_sizes: list) -> int:
    return len(levels) * n_patterns + (len(corner_sizes) if corners else 0) + len(occlusion_sizes) * n_patterns


def _nearest(values: list, target: float):
    """The element of `values` closest to `target` (first one on ties), or None when empty."""
    return min(values, key=lambda v: abs(float(v) - target)) if values else None


# --------------------------------------------------------------------------- plots


def _note(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#555555")
    ax.set_xticks([])
    ax.set_yticks([])


def _show(ax, matrix: np.ndarray, side: int, title: str = "", color: str | None = None) -> None:
    ax.imshow(np.asarray(matrix).reshape(side, side), cmap="binary", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(color or "#888888")
        spine.set_linewidth(2.4 if color else 0.8)


def _plot_patterns(images: list[Any], P: np.ndarray, side: int, path: Path) -> None:
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    axes = np.asarray(axes).reshape(2, n)
    for col, img in enumerate(images):
        black = float((P[col] == 1).mean())
        axes[0, col].imshow(img.pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0, col].set_title(f"{img.name}\nimagen original", fontsize=9)
        _show(axes[1, col], P[col], side, f"matriz ±1 (negro = +1)\nnegro {black:.3f}")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
    fig.suptitle(f"Los {n} códigos QR: imagen y matriz {side}×{side} que se almacena (N = {side * side})", fontsize=11)
    fig.tight_layout(h_pad=2.5)
    save_figure(fig, path)


def _plot_overlaps(ov: np.ndarray, names: list[str], mean_off: float, max_off: float, pair: str, path: Path) -> None:
    n = len(names)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(ov, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n), labels=names, rotation=30)
    ax.set_yticks(range(n), labels=names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{ov[i, j]:.2f}", ha="center", va="center", fontsize=10, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Solapamiento m = ξa · ξb / N")
    ax.set_title(
        f"Solapamiento entre códigos: promedio fuera de la diagonal {mean_off:.4f}\nmáximo {max_off:.4f} ({pair})",
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, path)


def _plot_shared(P: np.ndarray, side: int, shared: np.ndarray, path: Path) -> None:
    n_shared = int(shared.sum())
    n_total = shared.size
    rgb = np.zeros((n_total, 3))
    dark, light, orange = np.array([0.15] * 3), np.array([0.93] * 3), np.array([0.902, 0.624, 0.0])
    rgb[:] = orange
    black_all = shared & (P[0] == 1)
    white_all = shared & (P[0] == -1)
    rgb[black_all] = dark
    rgb[white_all] = light
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.0), gridspec_kw={"width_ratios": [1.5, 1]})
    ax1.imshow(rgb.reshape(side, side, 3), interpolation="nearest")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Módulos idénticos en todos los códigos (gris) y módulos que difieren (naranja)", fontsize=9)
    bars = ax2.bar(["idénticos\nen todos", "difieren\nen alguno"], [n_shared, n_total - n_shared], color=[ATTRACTOR_COLOR, SHARED_COLOR])
    for bar, value in zip(bars, [n_shared, n_total - n_shared], strict=True):
        ax2.text(bar.get_x() + bar.get_width() / 2, value, f"{value}\n({value / n_total:.4f})", ha="center", va="bottom", fontsize=10)
    ax2.set_ylim(0, n_total * 0.85)
    ax2.set_ylabel("Cantidad de módulos")
    ax2.set_title(f"{n_shared} de {n_total} módulos son compartidos ({n_shared / n_total:.1%})", fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)


def _plot_attractor(
    attractor: np.ndarray,
    P: np.ndarray,
    names: list[str],
    ov_attractor: list[float],
    unstable: np.ndarray,
    side: int,
    path: Path,
) -> None:
    n = len(names)
    fig, axes = plt.subplots(2, n + 1, figsize=(2.5 * (n + 1), 5.8))
    _show(axes[0, 0], attractor, side, "sign(Σ ξ): el «QR promedio»", color=ATTRACTOR_COLOR)
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.5, 0.5,
        "Abajo: en rojo, los módulos\ninestables de cada código\nbajo la regla de Hebb\n(el código no es punto fijo)",
        ha="center", va="center", fontsize=8.5, transform=axes[1, 0].transAxes,
    )  # fmt: skip
    for k, name in enumerate(names):
        _show(axes[0, k + 1], P[k], side, f"{name}\nm con el promedio = {ov_attractor[k]:.3f}")
        rgb = np.where((P[k] == 1)[:, None], 0.75, 0.97) * np.ones((1, 3))
        rgb[unstable[k]] = [0.835, 0.369, 0.0]
        axes[1, k + 1].imshow(rgb.reshape(side, side, 3), interpolation="nearest")
        axes[1, k + 1].set_xticks([])
        axes[1, k + 1].set_yticks([])
        axes[1, k + 1].set_title(f"{int(unstable[k].sum())} módulos inestables", fontsize=8)
    fig.suptitle("Con Hebb todas las consultas caen en el mismo estado espurio: el signo de la suma de los códigos", fontsize=10.5)
    fig.tight_layout()
    save_figure(fig, path)


def _result_label(info: dict[str, Any]) -> str:
    if info["recovered"]:
        return "recuperado"
    if info["is_mean_attractor"]:
        return "QR promedio"
    if info["final_class"] == "stored":
        return f"otro código ({info['final_pattern']})"
    if info["final_class"] == "inverted":
        return f"invertido ({info['final_pattern']})"
    return "estado espurio"


def _plot_examples(examples: list[dict[str, Any]], rules: list[str], side: int, path: Path) -> None:
    n_cols = 1 + len(rules)
    fig, axes = plt.subplots(len(examples), n_cols, figsize=(2.7 * n_cols, 2.9 * len(examples)))
    axes = np.asarray(axes).reshape(len(examples), n_cols)
    for row, ex in enumerate(examples):
        _show(axes[row, 0], ex["query"], side, ex["title"], color="#E69F00")
        for col, rule in enumerate(rules, start=1):
            info = ex["results"][rule]
            _show(
                axes[row, col], info["final"], side,
                f"{RULE_SHORT[rule]}: {_result_label(info)}\n{info['iterations']} iteraciones",
                color="#009E73" if info["recovered"] else RULE_COLORS[rule],
            )  # fmt: skip
    fig.suptitle(f"Ejemplos de consultas dañadas del código «{ex['pattern']}» y su estado final", fontsize=10.5)
    fig.tight_layout()
    save_figure(fig, path)


def _damage_curve(rates: list[dict[str, Any]], damage: str, rule: str) -> list[dict[str, Any]]:
    return [r for r in rates if r["damage"] == damage and r["rule"] == rule]


def _plot_recovery(rates: list[dict[str, Any]], rules: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), sharey=True)
    panels = (
        ("flip", "Ruido aleatorio (módulos invertidos)", "Nivel de ruido (fracción de módulos invertidos)"),
        ("torn_corner", "Esquina rota", "Fracción de módulos dañados (promedio por tamaño)"),
        ("occlusion", "Manchas oscuras", "Fracción de módulos dañados (promedio por tamaño)"),
    )
    for ax, (damage, title, xlabel) in zip(axes, panels, strict=True):
        ax.set_title(title, fontsize=10)
        if not any(r["damage"] == damage for r in rates):
            _note(ax, "sin consultas")
            continue
        for rule in rules:
            sub = _damage_curve(rates, damage, rule)
            x = [r["level"] if damage == "flip" else r["mean_damaged_frac"] for r in sub]
            ax.plot(x, [r["recovery_rate"] for r in sub], marker="o", linewidth=2.0, color=RULE_COLORS[rule], label=RULE_LABELS[rule])
            if rule == "hebb":
                ax.plot(
                    x, [r["mean_attractor_rate"] for r in sub], linestyle="--", marker="x", linewidth=1.5,
                    color=ATTRACTOR_COLOR, label="Hebb: termina en el «QR promedio»",
                )  # fmt: skip
        if damage != "flip":
            first = _damage_curve(rates, damage, rules[0])
            for r in first:
                ax.annotate(str(r["size"]), (r["mean_damaged_frac"], r["recovery_rate"]), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7, color="#555555")  # fmt: skip
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.04, 1.08)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Tasa de recuperación exacta")
    axes[0].legend(loc="center left", fontsize=8)
    fig.suptitle("Recuperación exacta de los códigos QR según el tipo de deterioro (los números son tamaños)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, path)


def _plot_decode(rates: list[dict[str, Any]], rules: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), sharey=True)
    panels = (
        ("flip", "Ruido aleatorio (módulos invertidos)", "Nivel de ruido (fracción de módulos invertidos)"),
        ("torn_corner", "Esquina rota", "Fracción de módulos dañados (promedio por tamaño)"),
        ("occlusion", "Manchas oscuras", "Fracción de módulos dañados (promedio por tamaño)"),
    )
    for ax, (damage, title, xlabel) in zip(axes, panels, strict=True):
        ax.set_title(title, fontsize=10)
        base = _damage_curve(rates, damage, rules[0])
        if not base:
            _note(ax, "sin consultas")
            continue
        x = [r["level"] if damage == "flip" else r["mean_damaged_frac"] for r in base]
        ax.plot(
            x, [r["query_decode_rate"] for r in base], linestyle="--", marker="s", linewidth=1.8, color="#555555",
            label="Consulta dañada: solo la corrección de errores del QR",
        )  # fmt: skip
        for rule in rules:
            sub = _damage_curve(rates, damage, rule)
            ax.plot(
                x, [r["final_decode_rate"] for r in sub], marker="o", linewidth=2.0, color=RULE_COLORS[rule],
                label=f"Estado recuperado: {RULE_LABELS[rule]}",
            )  # fmt: skip
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.04, 1.08)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Fracción que el lector decodifica como el menú original")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Lectura con zxing-cpp: consulta dañada frente a estado recuperado por la red de Hopfield", fontsize=11)
    fig.tight_layout()
    save_figure(fig, path)


def _plot_iterations(rates: list[dict[str, Any]], rules: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for rule in rules:
        sub = _damage_curve(rates, "flip", rule)
        x = [r["level"] for r in sub]
        ax.plot(x, [r["mean_iterations"] for r in sub], marker="o", color=RULE_COLORS[rule], linewidth=2.0, label=f"{RULE_SHORT[rule]}: promedio")
        ax.plot(x, [r["max_iterations"] for r in sub], marker="^", linestyle="--", color=RULE_COLORS[rule], linewidth=1.5, label=f"{RULE_SHORT[rule]}: máximo")
    ax.set_xlabel("Nivel de ruido (fracción de módulos invertidos)")
    ax.set_ylabel("Iteraciones hasta converger (barridos que cambian el estado)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Iteraciones de la recuperación asincrónica frente al ruido")
    fig.tight_layout()
    save_figure(fig, path)


def _plot_energy(
    trace_df: pd.DataFrame, seqs: list[tuple[dict[str, Any], np.ndarray]], rules: list[str], levels: list[float],
    n_neurons: int, path: Path,
) -> None:  # fmt: skip
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2))
    if trace_df.empty:
        _note(ax1, "sin trazas por actualización")
    else:
        styles = ["-", "--", ":", "-."]
        trace_levels = list(dict.fromkeys(trace_df["level"].tolist()))
        for (rule, level), group in trace_df.groupby(["rule", "level"], sort=False):
            ax1.plot(
                group["step"], group["energy"], color=RULE_COLORS[rule], linestyle=styles[trace_levels.index(level) % 4],
                linewidth=1.6, label=f"{RULE_SHORT[rule]}, ruido {level:g}",
            )  # fmt: skip
        max_step = int(trace_df["step"].max())
        for boundary in range(n_neurons, max_step + 1, n_neurons):
            ax1.axvline(boundary, color="#999999", linestyle=":", linewidth=0.8)
        ax1.set_xlabel(f"Actualizaciones de neuronas individuales (líneas punteadas = barridos de N = {n_neurons})")
        ax1.set_ylabel("Energía H(S)")
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=8)
    ax1.set_title("Energía tras cada actualización (una consulta por nivel y regla)", fontsize=10)
    cmap = plt.get_cmap("viridis")
    for rule in rules:
        for li, level in enumerate(levels):
            group = [s for row, s in seqs if row["rule"] == rule and row["damage"] == "flip" and row["level"] == level]
            if not group:
                continue
            length = max(len(s) for s in group)
            padded = np.stack([np.concatenate([s, np.full(length - len(s), s[-1])]) for s in group])
            ax2.plot(
                range(length), np.median(padded, axis=0), color=cmap(li / max(1, len(levels) - 1)),
                linestyle="-" if rule == "hebb" else "--", marker="o", markersize=3, linewidth=1.3,
            )  # fmt: skip
    ax2.set_xlabel("Iteración (barrido completo)")
    ax2.set_ylabel("Energía H(S), mediana de las consultas")
    ax2.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(levels), max(levels)))
    fig.colorbar(sm, ax=ax2).set_label("Nivel de ruido (color); Hebb continua, pseudoinversa discontinua")
    ax2.set_title("Energía por iteración, consultas de ruido aleatorio (mediana)", fontsize=10)
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
    corners = [c for c in CORNERS if c in p["corners"]]
    corner_sizes = [int(s) for s in p["torn_corner_sizes"]] if corners else []
    occlusion_sizes = [int(s) for s in p["occlusion_sizes"]]
    trace_levels = [float(x) for x in p["trace_levels"]]

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

    # QR reader (optional zxing-cpp). It is only reached through the module object, so tests can
    # replace it. A decode is correct only when the text equals the target original's text.
    decode_enabled = bool(p["decode"])
    decode_scale = int(p["decode_scale"])
    decode_quiet_zone = int(p["decode_quiet_zone"])
    status = qrmod.decoder_status() if decode_enabled else None
    decoding = bool(status is not None and status.available)
    decode_cache: dict[bytes, str | None] = {}

    def decode_state(vector: np.ndarray) -> str | None:
        key = np.ascontiguousarray(vector, dtype=np.int8).tobytes()
        if key not in decode_cache:
            decode_cache[key] = qrmod.decode_matrix(
                np.asarray(vector, dtype=np.int8).reshape(side, side), scale=decode_scale, quiet_zone=decode_quiet_zone
            )
        return decode_cache[key]

    reference_texts: list[str | None] = [decode_state(P[k]) for k in range(n_patterns)] if decoding else []

    def decoded_as(text: str | None) -> str:
        if text is None:
            return "none"
        for k, ref in enumerate(reference_texts):
            if ref is not None and ref == text:
                return names[k]
        return "other"

    def decode_ok(text: str | None, target: int) -> bool | None:
        ref = reference_texts[target] if decoding else None
        return None if ref is None else bool(text == ref)

    weights: dict[str, np.ndarray] = {}
    stored_rows: list[dict[str, Any]] = []
    rule_stats: dict[str, dict[str, Any]] = {}
    for rule in sorted({*rules, "hebb"}, key=("hebb", "pseudoinverse").index):
        try:
            W = train(P, rule)
        except ValueError as exc:
            raise DataError(f"hopfield_qr: cannot store the codes with rule {rule}: {exc}") from exc
        weights[rule] = W
        fixed = is_fixed_point(W, P)
        margins = stability_margins(W, P)
        unstable_mask = sign_keep(P.astype(np.float64) @ W, P) != P
        energies = np.asarray(energy(W, P), dtype=float)
        rule_stats[rule] = {
            "n_fixed": int(np.sum(fixed)),
            "unstable": [int(u) for u in unstable_mask.sum(axis=1)],
            "unstable_mask": unstable_mask,
        }
        if rule not in rules:
            continue
        for k, name in enumerate(names):
            stored_rows.append(
                {
                    "rule": rule,
                    "pattern": name,
                    "fixed_point": bool(fixed[k]),
                    "n_unstable_bits": int(unstable_mask[k].sum()),
                    "stability_margin": round(float(margins[k]), 4),
                    "energy": round(float(energies[k]), 4),
                }
            )
    stored_rows.sort(key=lambda r: rules.index(r["rule"]))
    write_csv(ctx.artifact("stored.csv"), pd.DataFrame(stored_rows))

    # Why the codes are hard to store: correlation and shared modules.
    ov_matrix = dot_matrix(P) / float(n_neurons)
    ov_df = pd.DataFrame(ov_matrix, columns=names)
    ov_df.insert(0, "pattern", names)
    write_csv(ctx.artifact("overlaps.csv"), ov_df)
    pairs = [(i, j) for i in range(n_patterns) for j in range(i + 1, n_patterns)]
    pair_values = np.array([ov_matrix[i, j] for i, j in pairs])
    best_pair = pairs[int(np.argmax(pair_values))]
    shared = np.all(P == P[0], axis=0)
    attractor_overlaps = [round(float(v), 4) for v in overlaps(mean_attractor, P)]

    # ctx.rng is not used. One spawn_seeds list is consumed in cell order (see _build_cells).
    seeds = spawn_seeds(ctx.seed, _count_cells(n_patterns, levels, corners, corner_sizes, occlusion_sizes))
    cells = _build_cells(P, side, levels, trials, corners, corner_sizes, occlusion_sizes, seeds)

    # Example queries for damage_examples.png: first code, trial 0.
    example_specs: list[tuple[str, Any, str | None]] = []
    for target in (0.2, 0.4):
        lv = _nearest(levels, target)
        if lv is not None and ("flip", lv, None) not in example_specs:
            example_specs.append(("flip", lv, None))
    if corners and corner_sizes:
        example_specs.append(("torn_corner", _nearest(corner_sizes, 20), "top_left" if "top_left" in corners else corners[0]))
    if occlusion_sizes:
        example_specs.append(("occlusion", _nearest(occlusion_sizes, 16), None))
    example_data: dict[tuple, dict[str, Any]] = {}

    query_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    energy_seqs: list[tuple[dict[str, Any], np.ndarray]] = []
    groups: dict[tuple[str, Any, str], list[dict[str, Any]]] = {}
    energy_never_increased = True
    query_id = 0
    for cell in cells:
        queries = cell["queries"]
        m = queries.shape[0]
        key_value = cell["level"] if cell["size"] is None else cell["size"]
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
            example_key = (cell["damage"], key_value, cell["corner"][i])
            is_example = target == 0 and cell["trial"][i] == 0 and example_key in example_specs
            if is_example:
                example_data[example_key] = {"query": queries[i].copy(), "n_changed": n_changed, "results": {}}
            query_ok = decode_ok(decode_state(queries[i]), target) if decoding else None
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
                    "query_decode_ok": query_ok,
                    "final_decode_ok": decode_ok(decode_state(final), target) if decoding else None,
                    "final_decoded_as": decoded_as(decode_state(final)) if decoding else None,
                }
                query_rows.append(row)
                groups.setdefault((cell["damage"], key_value, rule), []).append(row)
                energy_seqs.append((row, e_seq))
                for it_num, e_val in enumerate(e_seq):
                    energy_rows.append(
                        {"query_id": query_id, "rule": rule, "iteration": int(it_num), "energy": float(e_val)}
                    )
                if is_example:
                    example_data[example_key]["results"][rule] = {
                        "final": final.copy(),
                        "iterations": row["iterations"],
                        "recovered": recovered,
                        "final_class": cls,
                        "final_pattern": row["final_pattern"],
                        "is_mean_attractor": row["is_mean_attractor"],
                    }
            query_id += 1

    # Per-update energy traces: trial 0 of the first code at every trace level, re-run with
    # the cell's own seeds, so each trace is exactly that query of queries.csv.
    trace_rows: list[dict[str, Any]] = []
    for level in trace_levels:
        li = next(idx for idx, lv in enumerate(levels) if _same_level(lv, level))
        cell = cells[li * n_patterns]
        for r_idx, rule in enumerate(rules):
            res = recall(
                weights[rule],
                cell["queries"][0:1],
                HopfieldConfig(rule=rule, update=update, max_iter=max_iter),
                np.random.default_rng([cell["seed"], r_idx]),
                trace=True,
            )
            assert res.update_energies is not None
            trace = res.update_energies[0]
            batch_row = next(
                r for r in query_rows
                if r["rule"] == rule and r["damage"] == "flip" and r["level"] == cell["level"]
                and r["pattern"] == names[0] and r["trial"] == 0
            )  # fmt: skip
            if bool(np.any(np.diff(trace) > ENERGY_TOL)) or abs(float(trace[-1]) - batch_row["final_energy"]) > ENERGY_TOL:
                energy_never_increased = False
            for step, e_val in enumerate(trace):
                trace_rows.append(
                    {
                        "rule": rule,
                        "level": cell["level"],
                        "pattern": names[0],
                        "step": int(step),
                        "sweep": 0 if step == 0 else (step - 1) // n_neurons + 1,
                        "energy": float(e_val),
                    }
                )
    trace_df = pd.DataFrame(trace_rows, columns=TRACE_COLUMNS)
    write_csv(ctx.artifact("energy_trace.csv"), trace_df)

    queries_df = pd.DataFrame(query_rows)
    for col in ("size", "top", "left"):
        queries_df[col] = queries_df[col].astype("Int64")
    for col in ("query_decode_ok", "final_decode_ok"):
        queries_df[col] = queries_df[col].astype("boolean")
    write_csv(ctx.artifact("queries.csv"), queries_df)
    write_csv(ctx.artifact("energy.csv"), pd.DataFrame(energy_rows))

    rate_rows: list[dict[str, Any]] = []
    for (damage, key, rule), rows in groups.items():
        count = len(rows)

        def frac(pred, rows=rows, count=count) -> float:
            return round(sum(1 for r in rows if pred(r)) / count, 4)

        iters = [r["iterations"] for r in rows]

        def decode_rate(field: str, rows=rows) -> float | None:
            known = [r[field] for r in rows if r[field] is not None]
            return round(sum(known) / len(known), 4) if known else None

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
                "query_decode_rate": decode_rate("query_decode_ok"),
                "final_decode_rate": decode_rate("final_decode_ok"),
            }
        )
    rates_df = pd.DataFrame(rate_rows)
    rates_df["size"] = rates_df["size"].astype("Int64")
    write_csv(ctx.artifact("recovery_rates.csv"), rates_df)

    def curve(rule: str, damage: str, field: str = "recovery_rate") -> list[float]:
        return [r[field] for r in rate_rows if r["damage"] == damage and r["rule"] == rule]

    def max_full_level(rule: str, field: str = "recovery_rate") -> float | None:
        """Highest flip level L such that every level <= L has rate 1.0 (None when level 0 fails)."""
        best: float | None = None
        for lv, rate in zip(levels, curve(rule, "flip", field), strict=True):
            if rate != 1.0:
                break
            best = lv
        return best

    if ctx.plots:
        _plot_patterns(images, P, side, ctx.artifact("patterns.png"))
        _plot_overlaps(
            ov_matrix, names, round(float(pair_values.mean()), 4), round(float(pair_values.max()), 4),
            f"{names[best_pair[0]]}-{names[best_pair[1]]}", ctx.artifact("overlaps.png"),
        )  # fmt: skip
        _plot_shared(P, side, shared, ctx.artifact("shared_modules.png"))
        _plot_attractor(
            mean_attractor, P, names, attractor_overlaps, rule_stats["hebb"]["unstable_mask"], side,
            ctx.artifact("hebb_attractor.png"),
        )  # fmt: skip
        examples = []
        for spec in example_specs:
            data = example_data.get(spec)
            if data is None:
                continue
            damage, key, corner = spec
            title = {
                "flip": f"Consulta: ruido {key:g} ({data['n_changed']} módulos)",
                "torn_corner": f"Consulta: esquina rota {corner}, tamaño {key}\n({data['n_changed']} módulos)",
                "occlusion": f"Consulta: mancha {key}×{key}\n({data['n_changed']} módulos)",
            }[damage]
            examples.append({**data, "title": title, "pattern": names[0]})
        _plot_examples(examples, rules, side, ctx.artifact("damage_examples.png"))
        _plot_recovery(rate_rows, rules, ctx.artifact("recovery_vs_noise.png"))
        if decoding:
            _plot_decode(rate_rows, rules, ctx.artifact("decode_vs_noise.png"))
        _plot_iterations(rate_rows, rules, ctx.artifact("iterations.png"))
        _plot_energy(trace_df, energy_seqs, rules, levels, n_neurons, ctx.artifact("energy_per_step.png"))

    off_mean = round(float(pair_values.mean()), 4)
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
        "mean_offdiag_overlap": off_mean,
        "max_offdiag_overlap": round(float(pair_values.max()), 4),
        "max_overlap_pair": f"{names[best_pair[0]]}-{names[best_pair[1]]}",
        "n_shared_modules": int(shared.sum()),
        "shared_module_frac": round(float(shared.mean()), 4),
        "hebb_mean_attractor_overlaps": attractor_overlaps,
        "corners": corners,
        "torn_corner_sizes": corner_sizes,
        "occlusion_sizes": occlusion_sizes,
    }
    for rule in rules:
        rule_rows = [r for r in query_rows if r["rule"] == rule]
        summary[f"{rule}_n_fixed_points"] = rule_stats[rule]["n_fixed"]
        summary[f"{rule}_unstable_bits"] = rule_stats[rule]["unstable"]
        summary[f"{rule}_flip_recovery"] = curve(rule, "flip")
        summary[f"{rule}_corner_recovery"] = curve(rule, "torn_corner")
        summary[f"{rule}_occlusion_recovery"] = curve(rule, "occlusion")
        summary[f"{rule}_corner_mean_damaged_frac"] = curve(rule, "torn_corner", "mean_damaged_frac")
        summary[f"{rule}_occlusion_mean_damaged_frac"] = curve(rule, "occlusion", "mean_damaged_frac")
        summary[f"{rule}_mean_iterations"] = round(float(np.mean([r["iterations"] for r in rule_rows])), 4)
        summary[f"{rule}_max_level_full_recovery"] = max_full_level(rule)
        summary[f"{rule}_final_flip_decode_rates"] = curve(rule, "flip", "final_decode_rate") if decoding else []
        summary[f"{rule}_final_corner_decode_rates"] = (
            curve(rule, "torn_corner", "final_decode_rate") if decoding else []
        )
        summary[f"{rule}_final_occlusion_decode_rates"] = (
            curve(rule, "occlusion", "final_decode_rate") if decoding else []
        )
        summary[f"{rule}_max_level_full_decode"] = max_full_level(rule, "final_decode_rate") if decoding else None
    first_rule = rules[0]
    summary["decode_enabled"] = decode_enabled
    summary["decoder_available"] = status.available if status is not None else None
    summary["decoder_name"] = status.name if status is not None else None
    summary["decoder_version"] = status.version if status is not None else None
    summary["decoder_reason"] = status.reason if status is not None else "params.decode is false"
    summary["decode_scale"] = decode_scale if decoding else None
    summary["decode_quiet_zone"] = decode_quiet_zone if decoding else None
    summary["reference_texts"] = reference_texts
    summary["n_originals_decoded"] = sum(1 for t in reference_texts if t is not None) if decoding else None
    summary["query_flip_decode_rates"] = curve(first_rule, "flip", "query_decode_rate") if decoding else []
    summary["query_corner_decode_rates"] = curve(first_rule, "torn_corner", "query_decode_rate") if decoding else []
    summary["query_occlusion_decode_rates"] = curve(first_rule, "occlusion", "query_decode_rate") if decoding else []
    summary["query_max_level_full_decode"] = max_full_level(first_rule, "query_decode_rate") if decoding else None
    summary["mean_attractor_decoded_as"] = decoded_as(decode_state(mean_attractor)) if decoding else None
    hebb_rows = [r for r in query_rows if r["rule"] == "hebb"]
    summary["hebb_frac_mean_attractor"] = (
        round(sum(1 for r in hebb_rows if r["is_mean_attractor"]) / len(hebb_rows), 4) if hebb_rows else None
    )
    summary["max_iterations"] = int(max(r["iterations"] for r in query_rows))
    summary["n_max_iter"] = int(sum(1 for r in query_rows if r["status"] == "max_iter"))
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
