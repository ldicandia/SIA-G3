"""Independent noise and recall-order repetitions for fixed Hopfield memories."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .hopfield_eval import classify_states, flip_bits, n_flips_for
from ..models.hopfield import HopfieldConfig, energy, hebbian_weights, recall

METRICS = (
    "recovery_rate", "other_rate", "inverted_rate", "spurious_rate", "cycle_rate",
    "converged_spurious_rate", "max_iter_rate", "energy_increase_rate", "mean_iterations", "mean_initial_energy", "mean_final_energy",
    "query_decode_rate", "final_decode_rate",
)


def aggregate_repetitions(frame: pd.DataFrame, keys: list[str], metrics=METRICS) -> pd.DataFrame:
    records = []
    for values, group in frame.groupby(keys, sort=False):
        if not isinstance(values, tuple):
            values = (values,)
        record = dict(zip(keys, values))
        record["n_seeds"] = int(group["seed"].nunique())
        for metric in metrics:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=1))
        records.append(record)
    return pd.DataFrame(records)


def repeated_recall(
    conditions: Mapping[str, Sequence[np.ndarray]],
    levels: Sequence[float],
    trials: int,
    max_iter: int,
    seeds: Sequence[int],
    updates: Sequence[str],
    decode: Callable[[np.ndarray], str | None] | None = None,
    decode_condition: str = "joint",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    networks = {name: [(patterns, hebbian_weights(patterns)) for patterns in groups]
                for name, groups in conditions.items()}
    unique = {}
    for groups in conditions.values():
        for patterns in groups:
            for pattern in patterns:
                unique.setdefault(pattern.tobytes(), (len(unique), pattern))
    rows = []
    for seed in seeds:
        for level_index, level in enumerate(levels):
            queries = {}
            for key, (pattern_index, pattern) in unique.items():
                rng = np.random.default_rng(np.random.SeedSequence([seed, level_index, pattern_index]))
                queries[key] = flip_bits(pattern, n_flips_for(level, pattern.size), trials, rng)
            for condition_index, (name, groups) in enumerate(networks.items()):
                for update_index, update in enumerate(updates):
                    parts = []
                    for network_index, (patterns, weights) in enumerate(groups):
                        q = np.vstack([queries[pattern.tobytes()] for pattern in patterns])
                        targets = np.repeat(patterns, trials, axis=0)
                        rng = np.random.default_rng(np.random.SeedSequence(
                            [seed, level_index, condition_index, network_index, update_index, 1]))
                        result = recall(weights, q, HopfieldConfig(update=update, max_iter=max_iter), rng)
                        classes, _ = classify_states(result.states, patterns)
                        recovered = np.all(result.states == targets, axis=1)
                        part = {
                            "recovery_rate": recovered.astype(float),
                            "other_rate": ((classes == "stored") & ~recovered).astype(float),
                            "inverted_rate": (classes == "inverted").astype(float),
                            "spurious_rate": (classes == "spurious").astype(float),
                            "cycle_rate": (result.status == "cycle").astype(float),
                            "converged_spurious_rate": ((classes == "spurious") & (result.status == "converged")).astype(float),
                            "max_iter_rate": (result.status == "max_iter").astype(float),
                            "energy_increase_rate": np.asarray([np.any(np.diff(sequence) > 1e-9) for sequence in result.energies], dtype=float),
                            "mean_iterations": result.iterations,
                            "mean_initial_energy": np.asarray(energy(weights, q)),
                            "mean_final_energy": np.asarray([sequence[-1] for sequence in result.energies]),
                            "query_decode_rate": np.full(len(q), np.nan),
                            "final_decode_rate": np.full(len(q), np.nan),
                        }
                        if decode is not None and name == decode_condition:
                            for i, target in enumerate(targets):
                                expected = decode(target)
                                if expected is not None:
                                    part["query_decode_rate"][i] = decode(q[i]) == expected
                                    part["final_decode_rate"][i] = decode(result.states[i]) == expected
                        parts.append(pd.DataFrame(part))
                    values = pd.concat(parts, ignore_index=True)
                    rows.append({"seed": int(seed), "set": name, "update": update,
                                 "level": float(level), "n_flips": n_flips_for(level, targets.shape[1]),
                                 "n_queries": len(values),
                                 "max_iterations": int(values["mean_iterations"].max()),
                                 **{metric: float(values[metric].mean()) for metric in METRICS}})
    by_seed = pd.DataFrame(rows)
    return by_seed, aggregate_repetitions(by_seed, ["set", "update", "level"])
