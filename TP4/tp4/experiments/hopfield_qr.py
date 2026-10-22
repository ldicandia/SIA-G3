"""Five 37×37 QR patterns, Hebbian storage and random bit-flip queries (TP4 §2.1)."""

from __future__ import annotations

import math
from hashlib import sha256
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from ..analysis import qr as qrmod
from ..analysis.hopfield_eval import classify_states, dot_matrix, flip_bits, n_flips_for, render_rows
from ..analysis.qr import (
    DEFAULT_QUIET_ZONE, DEFAULT_SCALE, MAX_QUIET_ZONE, MAX_SCALE,
    load_qr_image, matrix_to_image, qr_name,
)
from ..analysis.hopfield_repetitions import repeated_recall, aggregate_repetitions
from ..errors import ConfigError
from ..models.hopfield import (
    MAX_ITER, UPDATES, HopfieldConfig, energy, is_fixed_point, recall, sign_keep, hebbian_weights,
)
from ..plotting import plt, save_figure
from ..runs import Experiment, RunContext, resolve_project_path, write_csv

SIDE = 37
EXTRA_KEYS = frozenset({"images"})
PARAM_DEFAULTS: dict[str, Any] = {
    "n_repeats": 30,
    "update": "async",
    "max_iter": 100,
    "noise_levels": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    "trials": 20,
    "trace_levels": [0.1, 0.3, 0.45],
    "decode": True,
    "decode_scale": DEFAULT_SCALE,
    "decode_quiet_zone": DEFAULT_QUIET_ZONE,
}


def validate(config: dict) -> None:
    images = config.get("images")
    if not isinstance(images, list) or len(images) != 5:
        raise ConfigError("hopfield_qr: images must contain the five original QR paths")
    if any(not isinstance(path, str) or not path.lower().endswith(".png") for path in images):
        raise ConfigError("hopfield_qr: every image must be a .png path")
    if len(set(images)) != 5 or len({qr_name(path) for path in images}) != 5:
        raise ConfigError("hopfield_qr: image names and paths must be distinct")
    for path in images:
        if not resolve_project_path(path).is_file():
            raise ConfigError(f"hopfield_qr: image not found: {path}")
    p = config["params"]
    if type(p["n_repeats"]) is not int or not 2 <= p["n_repeats"] <= 200:
        raise ConfigError("n_repeats must be an integer in [2, 200]")
    if p["update"] not in UPDATES:
        raise ConfigError("hopfield_qr: update must be async or sync")
    for name, lo, hi in (
        ("max_iter", 1, MAX_ITER), ("trials", 1, 200),
        ("decode_scale", 1, MAX_SCALE), ("decode_quiet_zone", 0, MAX_QUIET_ZONE),
    ):
        if type(p[name]) is not int or not lo <= p[name] <= hi:
            raise ConfigError(f"hopfield_qr: params.{name} must be an integer in [{lo}, {hi}]")
    levels = p["noise_levels"]
    if not isinstance(levels, list) or not 1 <= len(levels) <= 21:
        raise ConfigError("hopfield_qr: params.noise_levels must contain 1 to 21 levels")
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in levels):
        raise ConfigError("hopfield_qr: noise levels must be finite numbers in [0, 1]")
    if any(a >= b for a, b in zip(levels, levels[1:])):
        raise ConfigError("hopfield_qr: noise levels must be strictly increasing")
    trace = p["trace_levels"]
    if not isinstance(trace, list) or any(type(x) not in (int, float) or x not in levels for x in trace):
        raise ConfigError("hopfield_qr: trace_levels must be a subset of noise_levels")
    if len(set(trace)) != len(trace):
        raise ConfigError("hopfield_qr: trace_levels must not contain duplicates")
    if trace and p["update"] != "async":
        raise ConfigError("hopfield_qr: per-neuron energy traces require async updates")
    if type(p["decode"]) is not bool:
        raise ConfigError("hopfield_qr: params.decode must be boolean")


def _show(ax, state, side, title):
    ax.imshow(np.asarray(state).reshape(side, side), cmap="binary", vmin=-1, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _plots(ctx, patterns, names, overlaps, shared, repeated, traces, examples, spurious):
    joint = repeated[repeated["set"] == "joint"]
    single = repeated[repeated["set"] == "single"]
    count = int(joint.n_seeds.iloc[0])
    side = SIDE
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    for ax, pattern, name in zip(axes, patterns, names):
        _show(ax, pattern, side, name)
    fig.suptitle("Cinco patrones originales, sin alterar módulos")
    save_figure(fig, ctx.artifact("patterns.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(overlaps, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(5), labels=names, rotation=30)
    ax.set_yticks(range(5), labels=names)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{overlaps[i,j]:.2f}", ha="center", va="center")
    ax.set_title("Similitud: (ξᵘ·ξᵛ)/(‖ξᵘ‖ ‖ξᵛ‖), con ‖ξ‖=37")
    fig.colorbar(im, ax=ax)
    save_figure(fig, ctx.artifact("overlaps.png"))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(shared.reshape(side, side), cmap="Greys", vmin=0, vmax=1)
    ax.set_title(f"Módulos iguales en los cinco QR: {shared.sum()}/{shared.size}")
    ax.set_xticks([])
    ax.set_yticks([])
    save_figure(fig, ctx.artifact("shared_modules.png"))

    fig, axes = plt.subplots(len(examples), 3, figsize=(8, 2.8 * len(examples)), squeeze=False)
    for row, example in enumerate(examples):
        _show(axes[row, 0], patterns[0], side, f"Original: {names[0]}")
        _show(axes[row, 1], example["query"], side, f"Consulta: {example['n_changed']} bits invertidos")
        _show(axes[row, 2], example["final"], side, f"Final: {example['class']}, {example['iterations']} iter.")
    save_figure(fig, ctx.artifact("damage_examples.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(joint["level"], joint["recovery_rate_mean"], yerr=joint["recovery_rate_std"],
                fmt="o-", capsize=3, label="Cinco QR almacenados juntos")
    ax.errorbar(single["level"], single["recovery_rate_mean"], yerr=single["recovery_rate_std"],
                fmt="s--", capsize=3, label="Un QR por red (control)")
    ax.set_xlabel("Fracción solicitada de módulos invertidos")
    ax.set_ylabel("Proporción de consultas")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    ax.set_title(f"Recuperación exacta: media ± DE entre {count} semillas")
    save_figure(fig, ctx.artifact("recovery_vs_noise.png"))

    if joint["final_decode_rate_mean"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.errorbar(joint["level"], joint["query_decode_rate_mean"], yerr=joint["query_decode_rate_std"],
                    fmt="s--", capsize=3, label="Consulta sin Hopfield")
        ax.errorbar(joint["level"], joint["final_decode_rate_mean"], yerr=joint["final_decode_rate_std"],
                    fmt="o-", capsize=3, label="Salida de Hopfield")
        ax.set_title(f"Media ± DE entre {count} semillas")
        ax.set_xlabel("Fracción solicitada de módulos invertidos")
        ax.set_ylabel("Proporción decodificada con la URL objetivo")
        ax.set_ylim(-0.03, 1.03)
        ax.legend()
        save_figure(fig, ctx.artifact("decode_vs_noise.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, rows in [("Cinco QR juntos", joint), ("Un QR por red", single)]:
        ax.errorbar(rows["level"], rows["mean_iterations_mean"], yerr=rows["mean_iterations_std"],
                    marker="o", capsize=3, label=label)
    ax.set_title(f"Media ± DE entre {count} semillas")
    ax.set_xlabel("Fracción solicitada de módulos invertidos")
    ax.set_ylabel("Barridos que modifican el estado")
    ax.legend()
    save_figure(fig, ctx.artifact("iterations.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, rows in [("Cinco QR juntos", joint), ("Un QR por red", single)]:
        ax.errorbar(rows["level"], rows["mean_final_energy_mean"], yerr=rows["mean_final_energy_std"],
                    marker="o", capsize=3, label=label)
    ax.set_xlabel("Fracción solicitada de módulos invertidos")
    ax.set_ylabel("Energía final media por consulta")
    ax.set_title(f"Media ± DE entre {count} semillas; pesos distintos por condición")
    ax.legend()
    save_figure(fig, ctx.artifact("final_energy.png"))

    if not traces.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        for level, group in traces.groupby("level", sort=False):
            ax.plot(group["step"], group["energy"], label=f"Ruido {level:g}")
        ax.set_xlabel("Visitas individuales a neuronas, incluida comprobación final")
        ax.set_ylabel("H(S) = −½ SᵀWS")
        ax.set_title(f"Ejemplos de la semilla {ctx.seed}")
        ax.legend()
        save_figure(fig, ctx.artifact("energy_per_step.png"))

    if spurious is not None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        _show(axes[0], spurious["query"], side, "Consulta observada")
        _show(axes[1], spurious["final"], side, "Estado espúreo alcanzado")
        save_figure(fig, ctx.artifact("attractor.png"))


def _storage_controls(ctx, patterns, names):
    storage_rows = []
    for count in range(1, len(patterns) + 1):
        for indices in combinations(range(len(patterns)), count):
            stored = patterns[list(indices)]
            fixed = is_fixed_point(hebbian_weights(stored), stored)
            storage_rows.append({
                "patterns": ";".join(names[i] for i in indices), "n_stored": count,
                "n_fixed_points": int(fixed.sum()),
                "fixed_patterns": ";".join(names[i] for i, stable in zip(indices, fixed) if stable),
            })
    storage = pd.DataFrame(storage_rows)
    write_csv(ctx.artifact("storage_control.csv"), storage)

    rng = np.random.default_rng(ctx.seed)
    rows = []
    n = patterns.shape[1]
    for name, pattern in zip(names, patterns):
        weights = hebbian_weights(pattern[None, :])
        for level in ctx.params["noise_levels"]:
            flips = n_flips_for(level, n)
            queries = flip_bits(pattern, flips, ctx.params["trials"], rng)
            result = recall(weights, queries,
                            HopfieldConfig(update="async", max_iter=ctx.params["max_iter"]), rng)
            rows.append({
                "pattern": name, "level": level, "n_flips": flips,
                "actual_noise": flips / n, "n_queries": len(queries),
                "recovery_rate": float(np.all(result.states == pattern, axis=1).mean()),
            })
    single = pd.DataFrame(rows)
    write_csv(ctx.artifact("single_pattern_control.csv"), single)
    all_fixed = bool((storage.loc[storage["n_stored"] == 1, "n_fixed_points"] == 1).all())
    recovery = single.groupby("level", sort=False)["recovery_rate"].mean().tolist()
    return all_fixed, recovery


def run(ctx: RunContext) -> dict[str, Any]:
    p = ctx.params
    side = SIDE
    images = [load_qr_image(path, side=side) for path in ctx.config["images"]]
    names = [image.name for image in images]
    patterns = np.stack([image.matrix.ravel() for image in images])
    n = patterns.shape[1]
    weights = hebbian_weights(patterns)
    norms = np.linalg.norm(patterns.astype(float), axis=1)
    overlaps = dot_matrix(patterns) / np.outer(norms, norms)
    shared = np.all(patterns == patterns[0], axis=0)
    fixed = is_fixed_point(weights, patterns)
    unstable = (sign_keep(patterns @ weights, patterns) != patterns).sum(axis=1)
    dots = pd.DataFrame(overlaps, columns=names)
    dots.insert(0, "pattern", names)
    write_csv(ctx.artifact("overlaps.csv"), dots)
    pairwise = []
    raw_dots = dot_matrix(patterns)
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            differences = int(np.count_nonzero(patterns[i] != patterns[j]))
            pairwise.append({"pattern_a": names[i], "pattern_b": names[j],
                             "vector_size": n, "dot_product": int(raw_dots[i, j]),
                             "norm_a": norms[i], "norm_b": norms[j],
                             "normalized_dot": overlaps[i, j], "hamming_distance": differences,
                             "hamming_fraction": differences / n})
    write_csv(ctx.artifact("pairwise_similarity.csv"), pd.DataFrame(pairwise))
    write_csv(ctx.artifact("inputs.csv"), pd.DataFrame({
        "pattern": names, "path": list(ctx.config["images"]),
        "sha256": [sha256(image.path.read_bytes()).hexdigest() for image in images],
        "rows": [side] * len(images), "cols": [side] * len(images),
        "vector_size": [n] * len(images), "vector_norm": norms,
    }))
    write_csv(ctx.artifact("stored.csv"), pd.DataFrame({
        "pattern": names, "fixed_point": fixed, "unstable_bits": unstable,
        "energy": energy(weights, patterns),
    }))

    status = qrmod.decoder_status() if p["decode"] else None
    decoding = status is not None and status.available
    decode_cache: dict[bytes, str | None] = {}

    def decode(state):
        if not decoding:
            return None
        key = state.tobytes()
        if key not in decode_cache:
            decode_cache[key] = qrmod.decode_matrix(
                state.reshape(side, side), scale=p["decode_scale"], quiet_zone=p["decode_quiet_zone"],
            )
        return decode_cache[key]

    reference = [decode(pattern) for pattern in patterns]
    rows, energies, traces, examples = [], [], [], []
    spurious = None
    energy_never_increased = True
    config = HopfieldConfig(update=p["update"], max_iter=p["max_iter"])
    for level in p["noise_levels"]:
        k = n_flips_for(level, n)
        for target, pattern in enumerate(patterns):
            queries = flip_bits(pattern, k, p["trials"], ctx.rng)
            trace = target == 0 and level in p["trace_levels"]
            result = recall(weights, queries, config, ctx.rng, trace=trace)
            classes, indices = classify_states(result.states, patterns)
            for trial, final in enumerate(result.states):
                query_id = len(rows)
                sequence = result.energies[trial]
                if p["update"] == "async" and np.any(np.diff(sequence) > 1e-9):
                    energy_never_increased = False
                query_text, final_text = decode(queries[trial]), decode(final)
                expected = reference[target]
                rows.append({
                    "query_id": query_id, "level": level, "pattern": names[target], "trial": trial,
                    "n_changed": k, "damaged_frac": k / n,
                    "iterations": int(result.iterations[trial]), "status": result.status[trial],
                    "final_class": classes[trial],
                    "final_pattern": names[indices[trial]] if indices[trial] >= 0 else "",
                    "recovered": bool(np.array_equal(final, pattern)),
                    "hamming_target": int(np.sum(final != pattern)),
                    "final_energy": float(sequence[-1]),
                    "query_decode_ok": query_text == expected if expected is not None else None,
                    "final_decode_ok": final_text == expected if expected is not None else None,
                })
                energies.extend({"query_id": query_id, "iteration": t, "energy": float(e)} for t, e in enumerate(sequence))
                if target == 0 and trial == 0:
                    examples.append({
                        "level": level, "query": queries[trial].copy(), "final": final.copy(),
                        "n_changed": k, "class": classes[trial], "iterations": int(result.iterations[trial]),
                    })
                if spurious is None and classes[trial] == "spurious" and result.status[trial] == "converged":
                    spurious = {"query": queries[trial].copy(), "final": final.copy()}
            if trace:
                values = result.update_energies[0]
                energy_never_increased &= not bool(np.any(np.diff(values) > 1e-9))
                traces.extend({
                    "query_id": len(rows) - p["trials"], "level": level,
                    "step": step, "sweep": (step - 1) // n + 1 if step else 0, "energy": float(value),
                } for step, value in enumerate(values))
    queries_df = pd.DataFrame(rows)
    write_csv(ctx.artifact("queries.csv"), queries_df)
    write_csv(ctx.artifact("energy.csv"), pd.DataFrame(energies))
    trace_df = pd.DataFrame(traces, columns=["query_id", "level", "step", "sweep", "energy"])
    write_csv(ctx.artifact("energy_trace.csv"), trace_df)
    rates = []
    for level, group in queries_df.groupby("level", sort=False):
        rates.append({
            "level": level, "n_queries": len(group),
            "recovery_rate": float(group["recovered"].mean()),
            "spurious_rate": float((group["final_class"] == "spurious").mean()),
            "mean_iterations": float(group["iterations"].mean()),
            "max_iterations": int(group["iterations"].max()),
            "query_decode_rate": float(group["query_decode_ok"].dropna().mean()) if decoding else None,
            "final_decode_rate": float(group["final_decode_ok"].dropna().mean()) if decoding else None,
        })
    rates_df = pd.DataFrame(rates)
    write_csv(ctx.artifact("recovery_rates.csv"), rates_df)
    singleton_fixed, _ = _storage_controls(ctx, patterns, names)
    seeds = list(range(ctx.seed, ctx.seed + int(p["n_repeats"])))
    by_seed, repeated = repeated_recall(
        {"joint": [patterns], "single": [pattern[None, :] for pattern in patterns]},
        p["noise_levels"], p["trials"], p["max_iter"], seeds, [p["update"]],
        decode=decode if decoding else None)
    write_csv(ctx.artifact("recall_by_seed.csv"), by_seed)
    write_csv(ctx.artifact("recall_repetitions.csv"), repeated)
    joint = repeated[repeated["set"] == "joint"]
    single = repeated[repeated["set"] == "single"]
    if ctx.plots:
        _plots(ctx, patterns, names, overlaps, shared, repeated, trace_df, examples[::max(1, len(examples)//3)][:4], spurious)
    pair_values = overlaps[np.triu_indices(5, 1)]
    return {
        "n_patterns": 5, "n_neurons": n, "side": side, "pattern_names": names,
        "roundtrip_identical": all(np.array_equal(matrix_to_image(image.matrix), image.pixels) for image in images),
        "capacity_015": 0.15 * n, "update": p["update"],
        "noise_levels": p["noise_levels"], "trials": p["trials"],
        "n_repeats": len(seeds), "seeds": seeds, "example_seed": ctx.seed,
        "example_n_queries": len(rows), "n_queries_per_seed": len(rows),
        "n_queries": len(seeds) * len(rows),
        "mean_offdiag_overlap": round(float(pair_values.mean()), 4),
        "max_offdiag_overlap": round(float(pair_values.max()), 4),
        "n_shared_modules": int(shared.sum()), "shared_module_frac": round(float(shared.mean()), 4),
        "n_fixed_points": int(fixed.sum()), "unstable_bits": unstable.tolist(),
        "recovery": joint.recovery_rate_mean.tolist(), "recovery_std": joint.recovery_rate_std.tolist(),
        "singleton_all_fixed": singleton_fixed,
        "single_pattern_recovery": single.recovery_rate_mean.tolist(),
        "single_pattern_recovery_std": single.recovery_rate_std.tolist(),
        "mean_iterations": joint.mean_iterations_mean.tolist(),
        "mean_iterations_std": joint.mean_iterations_std.tolist(),
        "mean_final_energy": joint.mean_final_energy_mean.tolist(),
        "mean_final_energy_std": joint.mean_final_energy_std.tolist(),
        "max_iterations": int(by_seed.loc[by_seed["set"] == "joint", "max_iterations"].max()),
        "energy_never_increased": bool(energy_never_increased and (by_seed.energy_increase_rate == 0).all()) if p["update"] == "async" else None,
        "spurious_found": spurious is not None,
        "spurious_state": render_rows(spurious["final"], (side, side)) if spurious is not None else [],
        "decode_enabled": p["decode"], "decoder_available": status.available if status else None,
        "decoder_name": status.name if status else None, "decoder_version": status.version if status else None,
        "decoder_reason": status.reason if status else "params.decode is false",
        "reference_texts": reference, "n_originals_decoded": sum(x is not None for x in reference) if decoding else None,
        "query_decode_rates": joint.query_decode_rate_mean.tolist() if decoding else [],
        "query_decode_rates_std": joint.query_decode_rate_std.tolist() if decoding else [],
        "final_decode_rates": joint.final_decode_rate_mean.tolist() if decoding else [],
        "final_decode_rates_std": joint.final_decode_rate_std.tolist() if decoding else [],
    }


EXPERIMENT = Experiment(
    kind="hopfield_qr", run=run, extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS, validate=validate,
)
