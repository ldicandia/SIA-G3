"""Descriptive campaign decisions (assignment 1.4), without a learned ranking model."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..features import dominant_topic

RANKABLE_COLUMNS = ("profile_visits_per_1000", "saves_per_1000", "content_tutorial_pct", "followers")


def build_evidence(raw, pc1, pc2, oja_index, som_bmu, som_row, som_col):
    out = raw.copy().reset_index(drop=True)
    out["dominant_topic"] = dominant_topic(raw).to_numpy()
    for name, values in {"pc1": pc1, "pc2": pc2, "oja_index": oja_index,
                         "som_bmu": som_bmu, "som_row": som_row,
                         "som_col": som_col}.items():
        out[name] = np.asarray(values)
    return out


def score_client(evidence: pd.DataFrame, client: Mapping) -> pd.DataFrame:
    rows = evidence.loc[evidence[client["topic"]] >= client["min_share"]].copy()
    return rows.sort_values(client["order_by"], ascending=False, kind="stable")


def pick(evidence: pd.DataFrame, client: Mapping, n: int) -> pd.DataFrame:
    rows = score_client(evidence, client)
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} candidates, fewer than {n}")
    return rows.iloc[:n].copy()


def followers_top(evidence: pd.DataFrame, n: int, mask=None) -> pd.DataFrame:
    rows = evidence if mask is None else evidence.loc[np.asarray(mask, dtype=bool)]
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} creators, fewer than {n}")
    return rows.sort_values("followers", ascending=False, kind="stable").iloc[:n].copy()


def strategy_metrics(rows: pd.DataFrame, topic: str) -> dict[str, float]:
    return {
        "mean_topic_share": float(rows[topic].mean()),
        "mean_argentina_audience_pct": float(rows["argentina_audience_pct"].mean()),
        "summed_est_arg_reach": float(rows["estimated_argentina_reach"].sum()),
        "mean_non_follower_reach_pct": float(rows["non_follower_reach_pct"].mean()),
        "mean_saves_per_1000": float(rows["saves_per_1000"].mean()),
        "mean_content_tutorial_pct": float(rows["content_tutorial_pct"].mean()),
        "mean_profile_visits_per_1000": float(rows["profile_visits_per_1000"].mean()),
        "total_price_usd": float(rows["price_usd"].sum()),
        "mean_followers": float(rows["followers"].mean()),
    }


def cell_profiles(evidence: pd.DataFrame, client: Mapping) -> pd.DataFrame:
    candidates = score_client(evidence, client)
    records = []
    for neuron, rows in candidates.groupby("som_bmu", sort=True):
        records.append({"client": client["name"], "som_bmu": int(neuron),
                        "n_creators": len(rows), "creators": ";".join(rows["name"]),
                        **strategy_metrics(rows, client["topic"])})
    return pd.DataFrame(records)


def describe_rule(client: Mapping) -> str:
    return (f"{client['topic']} >= {client['min_share']}%; "
            f"orden descendente por {client['order_by']} (supuesto comercial)")
