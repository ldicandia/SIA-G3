"""Deterministic selection of one SOM configuration from a sweep table (SOM-02).

The rule has four stages and every threshold lives here, once:

  (A) occupancy     empty_neuron_frac <= MAX_EMPTY_FRAC
  (B) topology      topographic_error <= (min TE over A) + TE_SLACK
  (C) order         distance_preservation >= (max DP over B) - DP_SLACK
  (D) accuracy      lowest quantization_error; ties by fewer neurons, then fewer epochs,
                    then the earlier row of the sweep

Thresholds are inclusive (a value exactly on the line passes; comparisons carry TOL so
float noise like 0.020000000000000004 does not flip a result). Pure pandas/numpy, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_EMPTY_FRAC = 0.5
TE_SLACK = 0.02
DP_SLACK = 0.05
TOL = 1e-12
METRIC_COLUMNS = ("quantization_error", "topographic_error", "distance_preservation", "empty_neuron_frac")
STAGE_COLUMNS = ("stage_occupancy", "stage_topology", "stage_order")


@dataclass(frozen=True)
class Selection:
    """`table` is the input plus stage_* flags, candidate_rank and selected; rows keep their order."""

    table: pd.DataFrame
    selected_position: int
    occupancy_fallback: bool


def rule_description() -> str:
    return (
        f"Se descartan los mapas con más de {MAX_EMPTY_FRAC:g} de neuronas vacías; entre los restantes se "
        f"conservan los de error topográfico a lo sumo {TE_SLACK:g} por encima del mejor y los de preservación "
        f"de distancias a lo sumo {DP_SLACK:g} por debajo de la mejor; de esos se elige el de menor error de "
        "cuantización (empates: menos neuronas, luego menos épocas, luego el primero del barrido)."
    )


def select_configuration(table: pd.DataFrame) -> Selection:
    if len(table) == 0:
        raise ValueError("select_configuration: the sweep table has no rows")
    missing = [c for c in (*METRIC_COLUMNS, "n_neurons", "epochs") if c not in table.columns]
    if missing:
        raise ValueError(f"select_configuration: the sweep table lacks columns {missing}")

    out = table.copy().reset_index(drop=True)
    for column in (*METRIC_COLUMNS, "n_neurons", "epochs"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    n = len(out)

    # (A) occupancy; when nothing passes every row stays a candidate and the fallback is recorded.
    occupancy = (out["empty_neuron_frac"] <= MAX_EMPTY_FRAC + TOL).to_numpy()
    fallback = not occupancy.any()
    if fallback:
        occupancy = np.ones(n, dtype=bool)

    # (B) topographic error within TE_SLACK of the best among (A).
    te = out["topographic_error"].to_numpy(dtype=float)
    te_ok = occupancy & np.isfinite(te)
    if te_ok.any():
        topology = te_ok & (te <= te[te_ok].min() + TE_SLACK + TOL)
    else:  # no usable TE anywhere: the stage cannot discriminate
        topology = occupancy.copy()

    # (C) distance preservation within DP_SLACK of the best among (B); NaN fails unless all NaN.
    dp = out["distance_preservation"].to_numpy(dtype=float)
    dp_ok = topology & np.isfinite(dp)
    if dp_ok.any():
        order = dp_ok & (dp >= dp[dp_ok].max() - DP_SLACK - TOL)
    else:
        order = topology.copy()

    # (D) stable lexicographic order: QE, n_neurons, epochs, original position.
    candidates = np.flatnonzero(order)
    qe = out["quantization_error"].to_numpy(dtype=float)
    neurons = out["n_neurons"].to_numpy(dtype=float)
    epochs = out["epochs"].to_numpy(dtype=float)

    def finite_last(values: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(values), values, np.inf)

    ranked = candidates[
        np.lexsort(
            (
                candidates,
                finite_last(epochs[candidates]),
                finite_last(neurons[candidates]),
                finite_last(qe[candidates]),
            )
        )
    ]
    rank = np.full(n, np.nan)
    rank[ranked] = np.arange(1, len(ranked) + 1)
    selected = np.zeros(n, dtype=bool)
    selected[ranked[0]] = True

    out["stage_occupancy"] = occupancy
    out["stage_topology"] = topology
    out["stage_order"] = order
    out["candidate_rank"] = pd.array(np.where(np.isnan(rank), -1, rank).astype(int), dtype="Int64")
    out.loc[np.isnan(rank), "candidate_rank"] = pd.NA
    out["selected"] = selected
    return Selection(table=out, selected_position=int(ranked[0]), occupancy_fallback=bool(fallback))
