"""Experiment kind `hopfield_letters`: Hopfield associative memory on 5x5 letters (Ej. 2.2).

Assignment items (docs/SIATP42026.pdf page 6, section 2.2):
1. Build 5x5 letter patterns with +1 ('#', black pixel) and -1 ('.', white pixel), store 4
   patterns and associate 5x5 queries with them using a hand-written Hopfield network.
2. Analyze pairwise orthogonality (dot-product matrix) and storage capacity (~0.138 * N and
   the course's 0.15 * N).
3. Enter a very noisy pattern and identify a spurious state.

Course conventions (clases/4.2 - Modelo de Hopfield.pdf):
- States in {-1, +1}; local field h_i = sum_{j != i} w_ij S_j; when h_i == 0 (|h_i| <= FIELD_TOL)
  the neuron keeps its previous state (slides 14-15).
- Hebbian weights w_ij = (1/N) sum_mu xi_i^mu xi_j^mu for i != j and w_ii = 0 (slides 18, 20).
- Energy H(S) = -1/2 sum_{i,j} w_ij S_i S_j (slides 34-36).

Config keys (besides common keys):
  alphabet   path to the JSON alphabet file (datos/letras_5x5.json)

Artifacts:
  dot_products.csv, orthogonality.csv, stored.csv, recall.csv, recall_rates.csv,
  energy.csv, energy_trace.csv, mixtures.csv, capacity.csv,
  alphabet.png, dot_products.png, capacity.png, recovery_vs_noise.png,
  recall_examples.png, spurious.png, energy.png, summary.json, run.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ..analysis.hopfield_eval import (
    classify_states,
    dot_matrix,
    flip_bits,
    mixture_states,
    n_flips_for,
    nearest_pattern,
    overlaps,
    rank_subsets,
    render_rows,
    stability_margins,
)
from ..errors import ConfigError
from ..models.hopfield import (
    MAX_ITER,
    RULES,
    UPDATES,
    HopfieldConfig,
    energy,
    is_fixed_point,
    recall,
    train,
)
from ..parallel import spawn_seeds
from ..plotting import grid_shape, plt, save_figure
from ..runs import Experiment, RunContext, resolve_project_path, write_csv

EXTRA_KEYS = frozenset({"alphabet"})
PARAM_DEFAULTS: dict[str, Any] = {
    "n_store": 4,
    "letters": [],
    "rule": "hebb",
    "updates": ["async", "sync"],
    "max_iter": 100,
    "noise_levels": [0.04, 0.08, 0.12, 0.16, 0.2, 0.28, 0.36, 0.44],
    "example_level": 0.2,
    "trials": 50,
    "spurious_noise": 0.4,
    "capacity_max_p": 10,
    "capacity_subsets": 200,
}
MAX_STORE = 6
MAX_SUBSETS = 300000
SELECTION_RULE = (
    "Menor máximo |producto interno| fuera de la diagonal, luego menor promedio |producto interno|, "
    "luego orden alfabético."
)


def _is_int(v: Any) -> bool:
    return type(v) is int


def _is_real(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def load_alphabet(path: str | Path) -> tuple[list[str], np.ndarray, tuple[int, int]]:
    """Load the JSON alphabet file using json.load only; return (letters, patterns, shape)."""
    p = resolve_project_path(path)
    try:
        with p.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"hopfield_letters: alphabet could not be read from {p}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise ConfigError("hopfield_letters: alphabet root must be a non-empty JSON object")

    for k in raw:
        if not (isinstance(k, str) and len(k) == 1 and "A" <= k <= "Z"):
            raise ConfigError(
                f"hopfield_letters: alphabet key must be a single uppercase ASCII letter, got {k!r}"
            )

    letters = sorted(raw.keys())
    shape: tuple[int, int] | None = None
    rows_list: list[np.ndarray] = []
    seen_patterns: dict[bytes, str] = {}

    for letter in letters:
        glyph = raw[letter]
        if not isinstance(glyph, list) or not glyph or not all(isinstance(r, str) and r for r in glyph):
            raise ConfigError(
                f"hopfield_letters: alphabet glyph for {letter!r} must be a non-empty list of strings"
            )
        n_rows = len(glyph)
        n_cols = len(glyph[0])
        if any(len(r) != n_cols for r in glyph):
            raise ConfigError(f"hopfield_letters: alphabet glyph for {letter!r} has unequal row lengths")
        if n_rows * n_cols < 2:
            raise ConfigError(f"hopfield_letters: alphabet glyph for {letter!r} must have at least 2 pixels")
        if shape is None:
            shape = (n_rows, n_cols)
        elif (n_rows, n_cols) != shape:
            raise ConfigError(
                f"hopfield_letters: alphabet glyph for {letter!r} has shape {(n_rows, n_cols)}, expected {shape}"
            )
        flat: list[int] = []
        for row_str in glyph:
            for ch in row_str:
                if ch == "#":
                    flat.append(1)
                elif ch == ".":
                    flat.append(-1)
                else:
                    raise ConfigError(
                        f"hopfield_letters: alphabet glyph for {letter!r} has invalid character {ch!r} "
                        "(only '#' and '.' allowed)"
                    )
        arr = np.asarray(flat, dtype=np.int8)
        key_bytes = arr.tobytes()
        if key_bytes in seen_patterns:
            other = seen_patterns[key_bytes]
            raise ConfigError(
                f"hopfield_letters: alphabet letters {other!r} and {letter!r} have identical patterns"
            )
        seen_patterns[key_bytes] = letter
        rows_list.append(arr)

    assert shape is not None
    patterns = np.stack(rows_list, axis=0)
    return letters, patterns, shape


def validate(config: dict) -> None:
    alphabet_path = config.get("alphabet")
    if not isinstance(alphabet_path, str) or not alphabet_path:
        raise ConfigError(f"hopfield_letters: alphabet must be a non-empty string path, got {alphabet_path!r}")
    letters_all, _, _ = load_alphabet(alphabet_path)
    n_letters = len(letters_all)

    p = config["params"]
    n_store = p["n_store"]
    if not (_is_int(n_store) and 2 <= n_store <= min(MAX_STORE, n_letters)):
        raise ConfigError(
            f"hopfield_letters: params.n_store must be an integer in [2, {min(MAX_STORE, n_letters)}], got {n_store!r}"
        )
    if math.comb(n_letters, n_store) > MAX_SUBSETS:
        raise ConfigError(
            f"hopfield_letters: C({n_letters}, {n_store}) exceeds MAX_SUBSETS ({MAX_SUBSETS})"
        )

    manual = p["letters"]
    if not isinstance(manual, list):
        raise ConfigError("hopfield_letters: params.letters must be a list")
    if manual:
        if len(manual) != n_store:
            raise ConfigError(
                f"hopfield_letters: params.letters must have length n_store ({n_store}), got {len(manual)}"
            )
        if len(set(manual)) != len(manual) or any(not isinstance(x, str) or x not in letters_all for x in manual):
            raise ConfigError(
                f"hopfield_letters: params.letters must be {n_store} distinct letters from the alphabet, got {manual!r}"
            )

    if p["rule"] not in RULES:
        raise ConfigError(f"hopfield_letters: params.rule must be one of {list(RULES)}, got {p['rule']!r}")

    updates = p["updates"]
    if not isinstance(updates, list) or not updates:
        raise ConfigError("hopfield_letters: params.updates must be a non-empty list")
    if len(set(updates)) != len(updates) or any(u not in UPDATES for u in updates):
        raise ConfigError(
            f"hopfield_letters: params.updates must contain distinct values from {list(UPDATES)}, got {updates!r}"
        )

    max_iter = p["max_iter"]
    if not (_is_int(max_iter) and 1 <= max_iter <= MAX_ITER):
        raise ConfigError(
            f"hopfield_letters: params.max_iter must be an integer in [1, {MAX_ITER}], got {max_iter!r}"
        )

    levels = p["noise_levels"]
    if not isinstance(levels, list) or not levels:
        raise ConfigError("hopfield_letters: params.noise_levels must be a non-empty list")
    if not all(_is_real(lv) and 0.0 <= float(lv) <= 1.0 for lv in levels):
        raise ConfigError(
            f"hopfield_letters: params.noise_levels must contain numbers in [0, 1], got {levels!r}"
        )
    for a, b in zip(levels, levels[1:], strict=False):
        if not float(a) < float(b):
            raise ConfigError(
                f"hopfield_letters: params.noise_levels must be strictly increasing, got {levels!r}"
            )

    ex_level = p["example_level"]
    if not _is_real(ex_level) or not any(
        math.isclose(float(ex_level), float(lv), rel_tol=0.0, abs_tol=1e-12) for lv in levels
    ):
        raise ConfigError(
            f"hopfield_letters: params.example_level must be one of params.noise_levels ({levels}), got {ex_level!r}"
        )

    trials = p["trials"]
    if not (_is_int(trials) and 1 <= trials <= 1000):
        raise ConfigError(f"hopfield_letters: params.trials must be an integer in [1, 1000], got {trials!r}")

    sp_noise = p["spurious_noise"]
    if not (_is_real(sp_noise) and 0.0 < float(sp_noise) <= 1.0):
        raise ConfigError(
            f"hopfield_letters: params.spurious_noise must be a number in (0, 1], got {sp_noise!r}"
        )

    cap_max_p = p["capacity_max_p"]
    if not (_is_int(cap_max_p) and 1 <= cap_max_p <= n_letters):
        raise ConfigError(
            f"hopfield_letters: params.capacity_max_p must be an integer in [1, {n_letters}], got {cap_max_p!r}"
        )

    cap_subsets = p["capacity_subsets"]
    if not (_is_int(cap_subsets) and 1 <= cap_subsets <= 5000):
        raise ConfigError(
            f"hopfield_letters: params.capacity_subsets must be an integer in [1, 5000], got {cap_subsets!r}"
        )


# --------------------------------------------------------------------------- plotting helpers


def _draw_pattern(ax, pattern: np.ndarray, shape: tuple[int, int], title: str = "", frame_color: str | None = None):
    grid = pattern.reshape(shape)
    ax.imshow(grid, cmap="binary", vmin=-1, vmax=1, interpolation="nearest")
    n_rows, n_cols = shape
    for r in range(n_rows + 1):
        ax.axhline(r - 0.5, color="#CCCCCC", linewidth=0.5)
    for c in range(n_cols + 1):
        ax.axvline(c - 0.5, color="#CCCCCC", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold" if frame_color else "normal")
    for spine in ax.spines.values():
        if frame_color:
            spine.set_edgecolor(frame_color)
            spine.set_linewidth(2.4)
        else:
            spine.set_edgecolor("#888888")
            spine.set_linewidth(0.8)


def _plot_alphabet(
    letters: list[str],
    patterns: np.ndarray,
    shape: tuple[int, int],
    sel_letters: list[str],
    worst_letters: list[str],
    path: Path,
) -> None:
    nrows, ncols = grid_shape(len(letters), ncols=7)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.65))
    axes_flat = np.atleast_1d(axes).ravel()
    sel_set = set(sel_letters)
    worst_set = set(worst_letters)
    for idx, ch in enumerate(letters):
        ax = axes_flat[idx]
        if ch in sel_set:
            color = "#0072B2"
            tag = f"{ch} (elegida)"
        elif ch in worst_set:
            color = "#D55E00"
            tag = f"{ch} (peor)"
        else:
            color = None
            tag = ch
        _draw_pattern(ax, patterns[idx], shape, title=tag, frame_color=color)
    for idx in range(len(letters), axes_flat.size):
        axes_flat[idx].axis("off")
    fig.suptitle(
        f"Abecedario 5×5 (negro = +1, blanco = -1) — Elegidas: {', '.join(sel_letters)} | Peores: {', '.join(worst_letters)}",
        fontsize=11,
    )
    fig.tight_layout()
    save_figure(fig, path)


def _plot_dot_products(
    letters: list[str],
    dots: np.ndarray,
    sel_letters: list[str],
    path: Path,
) -> None:
    n = len(letters)
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    im = ax.imshow(dots, cmap="coolwarm", vmin=-25, vmax=25)
    ax.set_xticks(np.arange(n), labels=letters, fontsize=8)
    ax.set_yticks(np.arange(n), labels=letters, fontsize=8)
    for i in range(n):
        for j in range(n):
            val = int(dots[i, j])
            text_color = "white" if abs(val) >= 15 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=6.5, color=text_color)
    sel_indices = [letters.index(ch) for ch in sel_letters]
    for idx in sel_indices:
        ax.add_patch(Rectangle((-0.5, idx - 0.5), n, 1, fill=False, edgecolor="#0072B2", linewidth=1.6))
        ax.add_patch(Rectangle((idx - 0.5, -0.5), 1, n, fill=False, edgecolor="#0072B2", linewidth=1.6))
    for i in sel_indices:
        for j in sel_indices:
            if i != j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#009E73", linewidth=2.2))
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Producto interno ξa · ξb")
    ax.set_title(
        f"Matriz de productos internos (26×26) — Conjunto más ortogonal resaltado: {', '.join(sel_letters)} (|dot| = 1)",
        fontsize=10.5,
    )
    fig.tight_layout()
    save_figure(fig, path)


def _plot_capacity(
    cap_df: pd.DataFrame,
    n_neurons: int,
    n_store: int,
    sel_letters: list[str],
    worst_letters: list[str],
    path: Path,
) -> None:
    rand_df = cap_df[cap_df["kind"] == "random"]
    sel_row = cap_df[cap_df["kind"] == "selected"].iloc[0]
    worst_row = cap_df[cap_df["kind"] == "worst"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        rand_df["p"],
        rand_df["frac_stable"],
        marker="o",
        color="#0072B2",
        linewidth=2.0,
        label="Fracción de letras estables (subconjuntos al azar)",
    )
    ax.plot(
        rand_df["p"],
        rand_df["frac_sets_all_stable"],
        marker="s",
        linestyle="--",
        color="#009E73",
        linewidth=1.8,
        label="Fracción de subconjuntos con todas las letras estables",
    )
    ax.scatter(
        [n_store],
        [sel_row["frac_stable"]],
        color="#0072B2",
        s=95,
        zorder=5,
        edgecolor="black",
        label=f"Elegidas ({','.join(sel_letters)}): {sel_row['frac_stable']:.2f}",
    )
    ax.scatter(
        [n_store],
        [worst_row["frac_stable"]],
        color="#D55E00",
        s=95,
        marker="D",
        zorder=5,
        edgecolor="black",
        label=f"Peores ({','.join(worst_letters)}): {worst_row['frac_stable']:.2f}",
    )
    c_0138 = 0.138 * n_neurons
    c_015 = 0.15 * n_neurons
    ax.axvline(c_0138, color="#CC79A7", linestyle=":", linewidth=1.8, label=f"0,138·N = {c_0138:.2f}")
    ax.axvline(c_015, color="#E69F00", linestyle="-.", linewidth=1.8, label=f"0,15·N = {c_015:.2f}")
    ax.set_xlabel("Cantidad de patrones almacenados (p)")
    ax.set_ylabel("Fracción de puntos fijos")
    ax.set_ylim(-0.04, 1.06)
    ax.set_xticks(rand_df["p"].tolist())
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Capacidad empírica de almacenamiento en función de p (N = 25)")
    fig.tight_layout()
    save_figure(fig, path)


def _plot_recovery_vs_noise(
    rates_df: pd.DataFrame,
    n_neurons: int,
    sel_letters: list[str],
    worst_letters: list[str],
    path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    styles = [
        ("selected", "async", "#0072B2", "-", "o", f"Elegidas ({','.join(sel_letters)}) — async"),
        ("selected", "sync", "#0072B2", "--", "s", f"Elegidas ({','.join(sel_letters)}) — sync"),
        ("worst", "async", "#D55E00", "-", "^", f"Peores ({','.join(worst_letters)}) — async"),
        ("worst", "sync", "#D55E00", "--", "v", f"Peores ({','.join(worst_letters)}) — sync"),
    ]
    for set_name, upd, color, ls, marker, label in styles:
        sub = rates_df[(rates_df["set"] == set_name) & (rates_df["update"] == upd)]
        if not sub.empty:
            ax1.plot(sub["level"], sub["recovery_rate"], color=color, linestyle=ls, marker=marker, linewidth=1.9, label=label)

    ax1.set_xlabel("Nivel de ruido (fracción de píxeles invertidos)")
    ax1.set_ylabel("Tasa de recuperación exacta")
    ax1.set_ylim(-0.04, 1.06)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_title("Recuperación frente al nivel de ruido")

    sel_async = rates_df[(rates_df["set"] == "selected") & (rates_df["update"] == "async")]
    if not sel_async.empty:
        x = np.arange(len(sel_async))
        labels_x = [f"{lv:g}\n({n_flips_for(lv, n_neurons)})" for lv in sel_async["level"]]
        rec = sel_async["recovery_rate"].to_numpy(dtype=float)
        oth = sel_async["other_rate"].to_numpy(dtype=float)
        inv = sel_async["inverted_rate"].to_numpy(dtype=float)
        spu = sel_async["spurious_rate"].to_numpy(dtype=float)

        ax2.bar(x, rec, color="#009E73", label="Recuperada (objetivo)")
        ax2.bar(x, oth, bottom=rec, color="#56B4E9", label="Otra letra almacenada")
        ax2.bar(x, inv, bottom=rec + oth, color="#E69F00", label="Letra invertida (−ξ)")
        ax2.bar(x, spu, bottom=rec + oth + inv, color="#CC79A7", label="Estado espúreo")
        ax2.set_xticks(x, labels=labels_x, fontsize=8)
        ax2.set_xlabel("Nivel de ruido (píxeles invertidos de 25)")
        ax2.set_ylabel("Proporción de estados finales")
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="lower left", fontsize=8)
        ax2.set_title(f"Destino de las consultas (elegidas {', '.join(sel_letters)}, async)")
        ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    save_figure(fig, path)


def _plot_recall_examples(
    examples: list[dict[str, Any]],
    shape: tuple[int, int],
    example_level: float,
    n_flips: int,
    path: Path,
) -> None:
    n_rows = len(examples)
    fig, axes = plt.subplots(n_rows, 3, figsize=(6.8, n_rows * 2.1))
    if n_rows == 1:
        axes = np.asarray([axes])
    for row_idx, ex in enumerate(examples):
        letter = ex["letter"]
        _draw_pattern(axes[row_idx, 0], ex["target"], shape, title=f"Original: {letter}")
        _draw_pattern(axes[row_idx, 1], ex["query"], shape, title=f"Consulta ({n_flips} flips)")
        status_lbl = "Recuperada" if ex["recovered"] else f"Final ({ex['final_class']})"
        frame = "#009E73" if ex["recovered"] else "#D55E00"
        _draw_pattern(
            axes[row_idx, 2],
            ex["final"],
            shape,
            title=f"{status_lbl} (it={ex['iterations']}, H={ex['final_energy']:.2f})",
            frame_color=frame,
        )
    fig.suptitle(
        f"Ejemplos de recuperación asincrónica (nivel de ruido {example_level:g}, {n_flips}/25 píxeles invertidos)",
        fontsize=10.5,
    )
    fig.tight_layout()
    save_figure(fig, path)


def _plot_spurious(
    sp_info: dict[str, Any],
    sel_letters: list[str],
    P_sel: np.ndarray,
    stored_energies: list[float],
    shape: tuple[int, int],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2 + len(sel_letters), figsize=(2.2 * (2 + len(sel_letters)), 3.2))
    _draw_pattern(
        axes[0],
        sp_info["query_vec"],
        shape,
        title=f"Consulta ({sp_info['letter']}, {sp_info['n_flips']} flips)",
        frame_color="#E69F00",
    )
    mix_lbl = f" [{sp_info['mixture']}]" if sp_info["mixture"] else ""
    _draw_pattern(
        axes[1],
        sp_info["state_vec"],
        shape,
        title=f"Espúreo{mix_lbl}\nH = {sp_info['energy']:.2f}",
        frame_color="#CC79A7",
    )
    for idx, ch in enumerate(sel_letters):
        ov = sp_info["overlaps"][idx]
        e_val = stored_energies[idx]
        _draw_pattern(
            axes[2 + idx],
            P_sel[idx],
            shape,
            title=f"Almacenada {ch}\nm={ov:+.2f}, H={e_val:.2f}",
        )
    fig.suptitle(
        f"Estado espúreo alcanzado desde una entrada muy ruidosa ({sp_info['n_flips']}/25 píxeles invertidos)",
        fontsize=10.5,
    )
    fig.tight_layout()
    save_figure(fig, path)


def _plot_energy_traces(
    trace_df: pd.DataFrame,
    n_neurons: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    palette = ["#0072B2", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#D55E00"]
    max_step = 0
    for idx, (ex_name, group) in enumerate(trace_df.groupby("example", sort=False)):
        steps = group["step"].to_numpy(dtype=int)
        vals = group["energy"].to_numpy(dtype=float)
        max_step = max(max_step, int(steps.max()))
        is_sp = str(ex_name).startswith("spurious")
        ax.plot(
            steps,
            vals,
            color=palette[idx % len(palette)],
            linestyle="--" if is_sp else "-",
            linewidth=2.0 if is_sp else 1.7,
            label=f"Espúreo ({ex_name})" if is_sp else f"Ejemplo {ex_name}",
        )
    for boundary in range(n_neurons, max_step + 1, n_neurons):
        ax.axvline(boundary, color="#888888", linestyle=":", linewidth=0.8)
    ax.set_xlabel(f" Visitas individuales a neuronas (líneas punteadas = barridos de N = {n_neurons})")
    ax.set_ylabel("Energía H(S)")
    ax.set_title("Evolución monótona no creciente de la energía H por actualización asincrónica")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, path)


# --------------------------------------------------------------------------- run


def run(ctx: RunContext) -> dict[str, Any]:
    letters, patterns, shape = load_alphabet(ctx.config["alphabet"])
    n_letters, n_neurons = patterns.shape
    p = ctx.params
    n_store = int(p["n_store"])
    rule = str(p["rule"])
    updates = [str(u) for u in p["updates"]]
    max_iter = int(p["max_iter"])
    levels = [float(x) for x in p["noise_levels"]]
    example_level = float(p["example_level"])
    trials = int(p["trials"])
    spurious_noise = float(p["spurious_noise"])
    capacity_max_p = int(p["capacity_max_p"])
    capacity_subsets = int(p["capacity_subsets"])

    dots = dot_matrix(patterns)
    dot_df = pd.DataFrame(dots, columns=letters)
    dot_df.insert(0, "letter", letters)
    write_csv(ctx.artifact("dot_products.csv"), dot_df)

    ranked = rank_subsets(dots, n_store)
    orth_rows = [
        {
            "rank": rank_idx,
            "letters": "".join(letters[i] for i in combo),
            "max_abs_dot": int(max_abs),
            "mean_abs_dot": round(float(mean_abs), 6),
        }
        for rank_idx, (combo, max_abs, mean_abs) in enumerate(ranked, start=1)
    ]
    write_csv(ctx.artifact("orthogonality.csv"), pd.DataFrame(orth_rows))

    if p["letters"]:
        sel_letters = list(p["letters"])
        sel_idx = tuple(letters.index(ch) for ch in sel_letters)
        off_vals = [abs(int(dots[a, b])) for i, a in enumerate(sel_idx) for b in sel_idx[i + 1 :]]
        sel_max_abs = max(off_vals)
        sel_mean_abs = sum(off_vals) / len(off_vals)
        selection = "manual"
    else:
        sel_idx, sel_max_abs, sel_mean_abs = ranked[0]
        sel_letters = [letters[i] for i in sel_idx]
        selection = "most_orthogonal"

    worst_idx, worst_max_abs, worst_mean_abs = ranked[-1]
    worst_letters = [letters[i] for i in worst_idx]

    P_sel = patterns[list(sel_idx)]
    P_worst = patterns[list(worst_idx)]
    W_sel = train(P_sel, rule)
    W_worst = train(P_worst, rule)

    fp_sel = is_fixed_point(W_sel, P_sel)
    fp_worst = is_fixed_point(W_worst, P_worst)
    margins_sel = [round(float(m), 4) for m in stability_margins(W_sel, P_sel)]
    margins_worst = [round(float(m), 4) for m in stability_margins(W_worst, P_worst)]
    energies_sel = [round(float(e), 4) for e in np.asarray(energy(W_sel, P_sel), dtype=float)]
    energies_worst = [round(float(e), 4) for e in np.asarray(energy(W_worst, P_worst), dtype=float)]

    stored_rows: list[dict[str, Any]] = []
    for set_name, set_letters, fps, margins, es in (
        ("selected", sel_letters, fp_sel, margins_sel, energies_sel),
        ("worst", worst_letters, fp_worst, margins_worst, energies_worst),
    ):
        for ch, fp, mg, ev in zip(set_letters, fps, margins, es, strict=True):
            stored_rows.append(
                {
                    "set": set_name,
                    "letter": ch,
                    "fixed_point": bool(fp),
                    "stability_margin": mg,
                    "energy": ev,
                }
            )
    write_csv(ctx.artifact("stored.csv"), pd.DataFrame(stored_rows))

    # ctx.rng is not used directly. One spawn_seeds list is consumed in fixed order:
    # 0: capacity, 1: spurious, then one per (set, letter, level) cell in order.
    sets_spec = [
        ("selected", sel_letters, P_sel, W_sel),
        ("worst", worst_letters, P_worst, W_worst),
    ]
    n_cells = sum(len(s_letters) * len(levels) for _, s_letters, _, _ in sets_spec)
    all_seeds = spawn_seeds(ctx.seed, 2 + n_cells)
    cap_seed = all_seeds[0]
    sp_seed = all_seeds[1]
    cell_seeds = all_seeds[2:]

    # 1. Capacity curve (consumes cap_seed)
    cap_rng = np.random.default_rng(cap_seed)
    cap_rows: list[dict[str, Any]] = []
    cap_p_list: list[int] = []
    cap_frac_stable_list: list[float] = []
    cap_frac_all_list: list[float] = []

    for p_val in range(1, capacity_max_p + 1):
        total_fp = 0
        sets_all_fp = 0
        for _ in range(capacity_subsets):
            sub_idx = cap_rng.choice(n_letters, size=p_val, replace=False)
            P_sub = patterns[sub_idx]
            W_sub = train(P_sub, rule)
            fps = is_fixed_point(W_sub, P_sub)
            n_ok = int(np.sum(fps))
            total_fp += n_ok
            if n_ok == p_val:
                sets_all_fp += 1
        f_stable = round(total_fp / (capacity_subsets * p_val), 4)
        f_all = round(sets_all_fp / capacity_subsets, 4)
        cap_rows.append(
            {
                "p": p_val,
                "kind": "random",
                "frac_stable": f_stable,
                "frac_sets_all_stable": f_all,
            }
        )
        cap_p_list.append(p_val)
        cap_frac_stable_list.append(f_stable)
        cap_frac_all_list.append(f_all)

    sel_n_fp = int(np.sum(fp_sel))
    worst_n_fp = int(np.sum(fp_worst))
    cap_rows.append(
        {
            "p": n_store,
            "kind": "selected",
            "frac_stable": round(sel_n_fp / n_store, 4),
            "frac_sets_all_stable": 1.0 if sel_n_fp == n_store else 0.0,
        }
    )
    cap_rows.append(
        {
            "p": n_store,
            "kind": "worst",
            "frac_stable": round(worst_n_fp / n_store, 4),
            "frac_sets_all_stable": 1.0 if worst_n_fp == n_store else 0.0,
        }
    )
    cap_df = pd.DataFrame(cap_rows)
    write_csv(ctx.artifact("capacity.csv"), cap_df)

    # 2. 3-letter mixtures of the selected set
    mix_order = min(3, n_store)
    mix_items = mixture_states(P_sel, order=mix_order)
    mix_rows: list[dict[str, Any]] = []
    n_stable_mixtures = 0
    for indices, signs, mix_state in mix_items:
        letters_str = "".join(sel_letters[i] for i in indices)
        signs_str = "".join(f"{'+' if s > 0 else '-'}{sel_letters[i]}" for i, s in zip(indices, signs, strict=True))
        stable = bool(is_fixed_point(W_sel, mix_state[np.newaxis, :])[0])
        cls_arr, _ = classify_states(mix_state[np.newaxis, :], P_sel)
        cls_str = str(cls_arr[0])
        if stable and cls_str == "spurious":
            n_stable_mixtures += 1
        mix_rows.append(
            {
                "letters": letters_str,
                "signs": signs_str,
                "stable": stable,
                "class": cls_str,
            }
        )
    write_csv(ctx.artifact("mixtures.csv"), pd.DataFrame(mix_rows))

    # 3. Spurious analysis at spurious_noise (consumes sp_seed)
    sp_rng = np.random.default_rng(sp_seed)
    sp_flips = n_flips_for(spurious_noise, n_neurons)
    sp_total_queries = 0
    sp_count = 0
    sp_inv_count = 0
    sp_found_info: dict[str, Any] | None = None

    for letter_pos, letter in enumerate(sel_letters):
        target = P_sel[letter_pos]
        q_sp = flip_bits(target, sp_flips, trials, sp_rng)
        res_sp = recall(
            W_sel,
            q_sp,
            HopfieldConfig(rule=rule, update="async", max_iter=max_iter),
            rng=sp_rng,
            trace=True,
            xp=ctx.backend.xp,
        )
        cls_sp, _ = classify_states(res_sp.states, P_sel)
        for t_idx in range(trials):
            sp_total_queries += 1
            c_str = str(cls_sp[t_idx])
            if c_str == "inverted":
                sp_inv_count += 1
            elif c_str == "spurious":
                sp_count += 1
                if sp_found_info is None and str(res_sp.status[t_idx]) == "converged":
                    st_vec = res_sp.states[t_idx]
                    ov_vec = overlaps(st_vec, P_sel)
                    near_idx = int(nearest_pattern(st_vec, P_sel)[0])
                    hamming_near = int(np.sum(st_vec != P_sel[near_idx]))
                    mix_label: str | None = None
                    for indices, signs, mix_state in mix_items:
                        if np.array_equal(st_vec, mix_state):
                            mix_label = "".join(
                                f"{'+' if s > 0 else '-'}{sel_letters[i]}"
                                for i, s in zip(indices, signs, strict=True)
                            )
                            break
                        if np.array_equal(st_vec, -mix_state):
                            mix_label = "".join(
                                f"{'-' if s > 0 else '+'}{sel_letters[i]}"
                                for i, s in zip(indices, signs, strict=True)
                            )
                            break
                    assert res_sp.update_energies is not None
                    sp_found_info = {
                        "letter": letter,
                        "trial": int(t_idx),
                        "n_flips": sp_flips,
                        "query_vec": q_sp[t_idx].copy(),
                        "state_vec": st_vec.copy(),
                        "query_rows": render_rows(q_sp[t_idx], shape),
                        "state_rows": render_rows(st_vec, shape),
                        "overlaps": [round(float(x), 4) for x in ov_vec],
                        "nearest_letter": sel_letters[near_idx],
                        "hamming_to_nearest": hamming_near,
                        "energy": round(float(res_sp.energies[t_idx][-1]), 4),
                        "mixture": mix_label,
                        "trace": res_sp.update_energies[t_idx].copy(),
                    }

    spurious_rate_val = round(sp_count / sp_total_queries, 4)
    inverted_rate_sp = round(sp_inv_count / sp_total_queries, 4)

    # 4. Recall sweep across sets, letters, levels, and updates (consumes cell_seeds)
    recall_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    examples_for_plot: list[dict[str, Any]] = []
    n_async_cycles = 0
    n_sync_cycles = 0
    energy_never_increased = True

    cell_cursor = 0
    for set_name, set_letters, P_set, W_set in sets_spec:
        for letter_pos, letter in enumerate(set_letters):
            target = P_set[letter_pos]
            for level in levels:
                cell_seed = cell_seeds[cell_cursor]
                cell_cursor += 1
                cell_rng = np.random.default_rng(cell_seed)
                k = n_flips_for(level, n_neurons)
                queries = flip_bits(target, k, trials, cell_rng)
                is_example_cell = set_name == "selected" and math.isclose(
                    level, example_level, rel_tol=0.0, abs_tol=1e-12
                )

                for upd in updates:
                    if upd == "async":
                        res = recall(
                            W_set,
                            queries,
                            HopfieldConfig(rule=rule, update="async", max_iter=max_iter),
                            rng=cell_rng,
                            trace=is_example_cell,
                            xp=ctx.backend.xp,
                        )
                    else:
                        res = recall(
                            W_set,
                            queries,
                            HopfieldConfig(rule=rule, update="sync", max_iter=max_iter),
                            xp=ctx.backend.xp,
                        )

                    cls_arr, idx_arr = classify_states(res.states, P_set)

                    if is_example_cell and upd == "async":
                        assert res.update_energies is not None
                        u_trace = res.update_energies[0]
                        for step_i, e_val in enumerate(u_trace):
                            trace_rows.append(
                                {
                                    "example": letter,
                                    "step": int(step_i),
                                    "energy": float(e_val),
                                }
                            )
                        examples_for_plot.append(
                            {
                                "letter": letter,
                                "target": target.copy(),
                                "query": queries[0].copy(),
                                "final": res.states[0].copy(),
                                "recovered": bool(np.array_equal(res.states[0], target)),
                                "final_class": str(cls_arr[0]),
                                "iterations": int(res.iterations[0]),
                                "final_energy": float(res.energies[0][-1]),
                            }
                        )

                    for t_idx in range(trials):
                        st_str = str(res.status[t_idx])
                        if st_str == "cycle":
                            if upd == "async":
                                n_async_cycles += 1
                            else:
                                n_sync_cycles += 1
                        c_str = str(cls_arr[t_idx])
                        m_idx = int(idx_arr[t_idx])
                        final_letter = set_letters[m_idx] if m_idx >= 0 else ""
                        final_state = res.states[t_idx]
                        rec = bool(np.array_equal(final_state, target))
                        hamming = int(np.sum(final_state != target))
                        e_seq = res.energies[t_idx]
                        if upd == "async" and e_seq.size > 1:
                            if bool(np.any(np.diff(e_seq) > 1e-9)):
                                energy_never_increased = False

                        recall_rows.append(
                            {
                                "set": set_name,
                                "update": upd,
                                "letter": letter,
                                "level": level,
                                "n_flips": k,
                                "trial": t_idx,
                                "iterations": int(res.iterations[t_idx]),
                                "status": st_str,
                                "final_class": c_str,
                                "final_letter": final_letter,
                                "recovered": rec,
                                "hamming_to_target": hamming,
                                "final_energy": float(e_seq[-1]),
                            }
                        )

                        qid = f"{set_name}_{upd}_{letter}_{level:g}_{t_idx}"
                        for it_num, e_val in enumerate(e_seq):
                            energy_rows.append(
                                {
                                    "query_id": qid,
                                    "iteration": int(it_num),
                                    "energy": float(e_val),
                                }
                            )

    if sp_found_info is not None:
        for step_i, e_val in enumerate(sp_found_info["trace"]):
            trace_rows.append(
                {
                    "example": f"spurious_{sp_found_info['letter']}",
                    "step": int(step_i),
                    "energy": float(e_val),
                }
            )

    recall_df = pd.DataFrame(recall_rows)
    write_csv(ctx.artifact("recall.csv"), recall_df)
    write_csv(ctx.artifact("energy.csv"), pd.DataFrame(energy_rows))
    trace_df = pd.DataFrame(trace_rows)
    write_csv(ctx.artifact("energy_trace.csv"), trace_df)

    # 5. Aggregate recall_rates.csv
    rate_rows: list[dict[str, Any]] = []
    for set_name, _, _, _ in sets_spec:
        for upd in updates:
            for level in levels:
                sub = recall_df[
                    (recall_df["set"] == set_name)
                    & (recall_df["update"] == upd)
                    & (np.isclose(recall_df["level"], level))
                ]
                n_q = len(sub)
                rec_r = round(float(sub["recovered"].mean()), 4)
                other_r = round(
                    float(((sub["final_class"] == "stored") & (~sub["recovered"])).mean()), 4
                )
                inv_r = round(float((sub["final_class"] == "inverted").mean()), 4)
                spu_r = round(float((sub["final_class"] == "spurious").mean()), 4)
                cyc_r = round(float((sub["status"] == "cycle").mean()), 4)
                max_r = round(float((sub["status"] == "max_iter").mean()), 4)
                mean_it = round(float(sub["iterations"].mean()), 4)
                rate_rows.append(
                    {
                        "set": set_name,
                        "update": upd,
                        "level": level,
                        "n_queries": int(n_q),
                        "recovery_rate": rec_r,
                        "other_rate": other_r,
                        "inverted_rate": inv_r,
                        "spurious_rate": spu_r,
                        "cycle_rate": cyc_r,
                        "max_iter_rate": max_r,
                        "mean_iterations": mean_it,
                    }
                )
    rates_df = pd.DataFrame(rate_rows)
    write_csv(ctx.artifact("recall_rates.csv"), rates_df)

    def _recovery_curve(s_name: str, u_name: str) -> list[float]:
        if u_name not in updates:
            return []
        sub = rates_df[(rates_df["set"] == s_name) & (rates_df["update"] == u_name)]
        return [round(float(x), 4) for x in sub["recovery_rate"].tolist()]

    sel_async_rec = _recovery_curve("selected", "async")
    sel_sync_rec = _recovery_curve("selected", "sync")
    worst_async_rec = _recovery_curve("worst", "async")
    worst_sync_rec = _recovery_curve("worst", "sync")

    max_full_level: float | None = None
    for lv, r_val in zip(levels, sel_async_rec, strict=False):
        if r_val == 1.0:
            max_full_level = lv
        else:
            break

    if ctx.plots:
        _plot_alphabet(letters, patterns, shape, sel_letters, worst_letters, ctx.artifact("alphabet.png"))
        _plot_dot_products(letters, dots, sel_letters, ctx.artifact("dot_products.png"))
        _plot_capacity(cap_df, n_neurons, n_store, sel_letters, worst_letters, ctx.artifact("capacity.png"))
        _plot_recovery_vs_noise(rates_df, n_neurons, sel_letters, worst_letters, ctx.artifact("recovery_vs_noise.png"))
        if examples_for_plot:
            _plot_recall_examples(
                examples_for_plot,
                shape,
                example_level,
                n_flips_for(example_level, n_neurons),
                ctx.artifact("recall_examples.png"),
            )
        if sp_found_info is not None:
            _plot_spurious(sp_found_info, sel_letters, P_sel, energies_sel, shape, ctx.artifact("spurious.png"))
        if not trace_df.empty:
            _plot_energy_traces(trace_df, n_neurons, ctx.artifact("energy.png"))

    cap_0138 = round(0.138 * n_neurons, 4)
    cap_015 = round(0.15 * n_neurons, 4)

    return {
        "n_neurons": int(n_neurons),
        "n_letters": int(n_letters),
        "n_store": int(n_store),
        "selection": selection,
        "selection_rule": SELECTION_RULE,
        "selected_letters": sel_letters,
        "selected_max_abs_dot": int(sel_max_abs),
        "selected_mean_abs_dot": round(float(sel_mean_abs), 4),
        "worst_letters": worst_letters,
        "worst_max_abs_dot": int(worst_max_abs),
        "worst_mean_abs_dot": round(float(worst_mean_abs), 4),
        "n_subsets": int(len(ranked)),
        "capacity_0138": cap_0138,
        "capacity_015": cap_015,
        "n_store_exceeds_0138": bool(n_store > cap_0138),
        "selected_n_fixed_points": sel_n_fp,
        "selected_stability_margins": margins_sel,
        "worst_n_fixed_points": worst_n_fp,
        "worst_stability_margins": margins_worst,
        "noise_levels": levels,
        "n_flips": [n_flips_for(lv, n_neurons) for lv in levels],
        "trials": trials,
        "selected_async_recovery": sel_async_rec,
        "selected_sync_recovery": sel_sync_rec,
        "worst_async_recovery": worst_async_rec,
        "worst_sync_recovery": worst_sync_rec,
        "selected_max_level_full_recovery": max_full_level,
        "n_async_cycles": int(n_async_cycles),
        "n_sync_cycles": int(n_sync_cycles),
        "energy_never_increased": bool(energy_never_increased),
        "spurious_noise": spurious_noise,
        "spurious_n_flips": int(sp_flips),
        "spurious_found": sp_found_info is not None,
        "spurious_letter": sp_found_info["letter"] if sp_found_info else None,
        "spurious_trial": sp_found_info["trial"] if sp_found_info else None,
        "spurious_rate": spurious_rate_val,
        "inverted_rate_at_spurious_noise": inverted_rate_sp,
        "spurious_query": sp_found_info["query_rows"] if sp_found_info else [],
        "spurious_state": sp_found_info["state_rows"] if sp_found_info else [],
        "spurious_overlaps": sp_found_info["overlaps"] if sp_found_info else [],
        "spurious_nearest_letter": sp_found_info["nearest_letter"] if sp_found_info else None,
        "spurious_hamming_to_nearest": sp_found_info["hamming_to_nearest"] if sp_found_info else None,
        "spurious_energy": sp_found_info["energy"] if sp_found_info else None,
        "stored_energies": energies_sel,
        "spurious_mixture": sp_found_info["mixture"] if sp_found_info else None,
        "n_stable_mixtures": int(n_stable_mixtures),
        "capacity_p": cap_p_list,
        "capacity_frac_stable": cap_frac_stable_list,
        "capacity_frac_sets_all_stable": cap_frac_all_list,
        "backend_resolved": ctx.backend.name,
    }


EXPERIMENT = Experiment(
    kind="hopfield_letters",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
