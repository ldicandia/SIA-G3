"""Experiment kind `hopfield_letters`: Hopfield associative memory on 5x5 letters (Ej. 2.2).

Assignment items (docs/SIATP42026.pdf page 6, section 2.2):
1. Build 5x5 letter patterns with +1 ('#', black pixel) and -1 ('.', white pixel), store 4
   patterns and associate 5x5 queries with them using a hand-written Hopfield network.
2. Analyze pairwise orthogonality (dot-product matrix) and storage capacity (~0.138 * N and
   the course's 0.15 * N).
3. Enter a very noisy pattern and identify a spurious state.

Course conventions (clases/4.2 - Modelo de Hopfield.pdf):
- States in {-1, +1}; local field h_i = sum_{j != i} w_ij S_j; when h_i == 0 the neuron keeps
  its previous state (slides 14-15).

Config keys (besides common keys):
  alphabet   path to the JSON alphabet file (e.g. datos/letras_5x5.json)

Artifacts (tracer):
  dot_products.csv, stored.csv, recall.csv, summary.json, run.json
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..errors import ConfigError
from ..models.hopfield import (
    MAX_ITER,
    RULES,
    UPDATES,
    HopfieldConfig,
    is_fixed_point,
    recall,
    train,
)
from ..parallel import spawn_seeds
from ..runs import Experiment, RunContext, resolve_project_path, write_csv

EXTRA_KEYS = frozenset({"alphabet"})
PARAM_DEFAULTS: dict[str, Any] = {
    "n_store": 4,
    "letters": [],
    "rule": "hebb",
    "updates": ["async"],
    "max_iter": 100,
    "noise_levels": [0.2],
    "trials": 50,
}
MAX_STORE = 6
MAX_SUBSETS = 300000


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

    trials = p["trials"]
    if not (_is_int(trials) and 1 <= trials <= 1000):
        raise ConfigError(f"hopfield_letters: params.trials must be an integer in [1, 1000], got {trials!r}")


def _rank_subsets_tracer(dots: np.ndarray, letters: list[str], k: int):
    best_key = None
    best_combo = None
    for combo in itertools.combinations(range(len(letters)), k):
        off_vals = [abs(int(dots[a, b])) for idx_a, a in enumerate(combo) for b in combo[idx_a + 1 :]]
        max_abs = max(off_vals)
        mean_abs = sum(off_vals) / len(off_vals)
        letter_tuple = tuple(letters[i] for i in combo)
        key = (max_abs, mean_abs, letter_tuple)
        if best_key is None or key < best_key:
            best_key = key
            best_combo = combo
    assert best_key is not None and best_combo is not None
    return best_combo, best_key[0], best_key[1], list(best_key[2])


def run(ctx: RunContext) -> dict[str, Any]:
    letters, patterns, _ = load_alphabet(ctx.config["alphabet"])
    n_letters, n_neurons = patterns.shape
    p = ctx.params
    n_store = int(p["n_store"])
    rule = str(p["rule"])
    max_iter = int(p["max_iter"])
    levels = [float(x) for x in p["noise_levels"]]
    trials = int(p["trials"])

    dots = patterns.astype(int) @ patterns.astype(int).T
    dot_df = pd.DataFrame(dots, columns=letters)
    dot_df.insert(0, "letter", letters)
    write_csv(ctx.artifact("dot_products.csv"), dot_df)

    if p["letters"]:
        sel_letters = list(p["letters"])
        sel_idx = tuple(letters.index(ch) for ch in sel_letters)
        off_vals = [abs(int(dots[a, b])) for idx_a, a in enumerate(sel_idx) for b in sel_idx[idx_a + 1 :]]
        sel_max_abs = max(off_vals)
        sel_mean_abs = sum(off_vals) / len(off_vals)
        selection = "manual"
    else:
        sel_idx, sel_max_abs, sel_mean_abs, sel_letters = _rank_subsets_tracer(dots, letters, n_store)
        selection = "most_orthogonal"

    P_sel = patterns[list(sel_idx)]
    W = train(P_sel, rule)
    fp_mask = is_fixed_point(W, P_sel)

    stored_rows = [
        {"set": "selected", "letter": ch, "fixed_point": bool(fp)}
        for ch, fp in zip(sel_letters, fp_mask, strict=True)
    ]
    write_csv(ctx.artifact("stored.csv"), pd.DataFrame(stored_rows))

    # ctx.rng is not used directly; spawn_seeds derives one deterministic seed per (letter, level) cell.
    n_cells = len(sel_letters) * len(levels)
    cell_seeds = spawn_seeds(ctx.seed, n_cells)

    recall_rows: list[dict[str, Any]] = []
    cell_idx = 0
    for letter_pos, letter in enumerate(sel_letters):
        target = P_sel[letter_pos]
        for level in levels:
            cell_rng = np.random.default_rng(cell_seeds[cell_idx])
            cell_idx += 1
            k = int(round(level * n_neurons))
            queries = np.repeat(target[np.newaxis, :], trials, axis=0)
            for t_idx in range(trials):
                if k > 0:
                    flip_pos = cell_rng.choice(n_neurons, size=k, replace=False)
                    queries[t_idx, flip_pos] = -queries[t_idx, flip_pos]
            res = recall(W, queries, HopfieldConfig(rule=rule, update="async", max_iter=max_iter), rng=cell_rng)
            for t_idx in range(trials):
                final_state = res.states[t_idx]
                rec = bool(np.array_equal(final_state, target))
                recall_rows.append(
                    {
                        "set": "selected",
                        "update": "async",
                        "letter": letter,
                        "level": level,
                        "n_flips": k,
                        "trial": t_idx,
                        "iterations": int(res.iterations[t_idx]),
                        "status": str(res.status[t_idx]),
                        "recovered": rec,
                        "final_energy": float(res.energies[t_idx][-1]),
                    }
                )

    write_csv(ctx.artifact("recall.csv"), pd.DataFrame(recall_rows))
    recovery_rate = round(sum(1 for r in recall_rows if r["recovered"]) / len(recall_rows), 4)
    cap_0138 = round(0.138 * n_neurons, 4)
    cap_015 = round(0.15 * n_neurons, 4)

    return {
        "n_neurons": int(n_neurons),
        "n_letters": int(n_letters),
        "n_store": int(n_store),
        "selection": selection,
        "selected_letters": sel_letters,
        "selected_max_abs_dot": int(sel_max_abs),
        "selected_mean_abs_dot": round(float(sel_mean_abs), 4),
        "selected_n_fixed_points": int(np.sum(fp_mask)),
        "capacity_0138": cap_0138,
        "capacity_015": cap_015,
        "n_store_exceeds_0138": bool(n_store > cap_0138),
        "recovery_rate": recovery_rate,
        "backend_resolved": ctx.backend.name,
    }


EXPERIMENT = Experiment(
    kind="hopfield_letters",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
