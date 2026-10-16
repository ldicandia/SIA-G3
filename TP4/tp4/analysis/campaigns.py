"""Campaign selection rule over the PCA, SOM and Oja evidence (Ej. 1.4), pure pandas/numpy.

D-10: no new model. A client's picks come from a transparent rule written in
`configs/campaigns.json`:

  1. a topic filter: the client's topic share must be >= `min_share` (inclusive; art uses the
     reconstructed `topic_art_pct`), and optionally the creator's SOM region must be in `regions`;
  2. a score, the sum of `weight * z(column)` over `rank_by`, where z is computed over the WHOLE
     catalog (ddof=0), not only over the candidates, so a score does not depend on the filter.

The picks are the top `n` candidates by score; equal scores keep the catalog (CSV) order.
Nothing here reads files, draws random numbers or looks at account names: picks are a function
of the evidence table and the client's rule only (prohibition CAMP-01).

The derived metrics are recomputed from raw columns (the feature set dropped them as exact
identities): estimated Argentine reach = average_reach * argentina_audience_pct / 100,
engagement per 1000 = likes + comments + saves + shares per 1000, and the cost per 1000
Argentine reach = price_usd / estimated Argentine reach * 1000.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..features import dominant_topic
from .som_maps import reconstruct_topic_art

DERIVED_COLUMNS = ("est_arg_reach", "engagement_per_1000", "cost_per_1000_arg_reach")

# price_ars and organic_posts_pct stay out: they are exact copies of price_usd / sponsored_posts_pct.
# The dropped total_engagement_per_1000 and estimated_argentina_reach are listed through their
# recomputed versions (engagement_per_1000, est_arg_reach).
RANKABLE_COLUMNS: tuple[str, ...] = (
    "followers",
    "average_reach",
    "average_impressions",
    "average_video_views",
    "monthly_posts",
    "argentina_audience_pct",
    "non_follower_reach_pct",
    "likes_per_1000",
    "comments_per_1000",
    "saves_per_1000",
    "shares_per_1000",
    "profile_visits_per_1000",
    "price_usd",
    "sponsored_posts_pct",
    "topic_football_pct",
    "topic_makeup_pct",
    "topic_dogs_pct",
    "topic_art_pct",
    "content_tutorial_pct",
    "content_review_pct",
    "content_entertainment_pct",
    "content_showcase_pct",
    "average_caption_words",
    "average_watch_time_seconds",
    "video_posts_pct",
    "short_form_posts_pct",
    "trend_posts_pct",
    "est_arg_reach",
    "engagement_per_1000",
    "cost_per_1000_arg_reach",
    "pc1",
    "pc2",
    "oja_index",
)

_ENGAGEMENT_PARTS = ("likes_per_1000", "comments_per_1000", "saves_per_1000", "shares_per_1000")


def add_derived(raw: pd.DataFrame) -> pd.DataFrame:
    """A copy of the raw creators frame with the three derived metrics and the reconstructed art share."""
    out = raw.copy()
    out["topic_art_pct"] = reconstruct_topic_art(raw).to_numpy()
    reach = raw["average_reach"] * raw["argentina_audience_pct"] / 100.0
    out["est_arg_reach"] = reach
    out["engagement_per_1000"] = raw[list(_ENGAGEMENT_PARTS)].sum(axis=1, skipna=False)
    out["cost_per_1000_arg_reach"] = raw["price_usd"] / reach.where(reach > 0) * 1000.0  # NaN where reach is 0
    return out


def build_evidence(
    raw: pd.DataFrame,
    pc1,
    pc2,
    oja_index,
    som_bmu,
    som_row,
    som_col,
    som_region,
) -> pd.DataFrame:
    """One row per creator in CSV order: ids, every rankable column, the SOM cell and the Oja rank."""
    derived = add_derived(raw)
    out = pd.DataFrame(
        {
            "creator_id": raw["creator_id"].to_numpy(),
            "name": raw["name"].to_numpy(),
            "dominant_topic": dominant_topic(raw).to_numpy(),
        }
    )
    extra = {"pc1": pc1, "pc2": pc2, "oja_index": oja_index}
    for column in RANKABLE_COLUMNS:
        out[column] = np.asarray(extra[column], dtype=float) if column in extra else derived[column].to_numpy()
    out["som_bmu"] = np.asarray(som_bmu, dtype=int)
    out["som_row"] = np.asarray(som_row, dtype=int)
    out["som_col"] = np.asarray(som_col, dtype=int)
    out["som_region"] = np.asarray(som_region, dtype=int)
    rank = np.empty(len(out), dtype=int)
    rank[np.argsort(-out["oja_index"].to_numpy(), kind="stable")] = np.arange(1, len(out) + 1)
    out["oja_rank"] = rank
    return out


def catalog_zscores(evidence: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """z-scores (ddof=0) of `columns` over every row of the evidence; ValueError on NaN or zero spread."""
    z = {}
    for column in columns:
        values = evidence[column].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError(f"cannot z-score {column!r}: it has NaN values")
        std = values.std(ddof=0)
        if not std > 0:
            raise ValueError(f"cannot z-score {column!r}: zero spread over the catalog")
        z[column] = (values - values.mean()) / std
    return pd.DataFrame(z, index=evidence.index)


def _candidate_mask(evidence: pd.DataFrame, client: Mapping) -> np.ndarray:
    mask = evidence[client["topic"]].to_numpy(dtype=float) >= float(client["min_share"])
    regions = client.get("regions") or []
    if regions:
        mask &= evidence["som_region"].isin(list(regions)).to_numpy()
    return mask


def rule_scores(evidence: pd.DataFrame, client: Mapping) -> np.ndarray:
    """The client's score of EVERY creator: sum of weight * z(column), z over the whole catalog (ddof=0)."""
    rank_by = client["rank_by"]
    z = catalog_zscores(evidence, [r["column"] for r in rank_by])
    score = np.zeros(len(evidence))
    for r in rank_by:
        score = score + float(r["weight"]) * z[r["column"]].to_numpy()
    return score


def score_client(evidence: pd.DataFrame, client: Mapping) -> pd.DataFrame:
    """The client's candidates with `topic_share` and `score`, best first (ties keep catalog order)."""
    score = rule_scores(evidence, client)  # over the FULL catalog, not only the candidates
    mask = _candidate_mask(evidence, client)
    candidates = evidence.loc[mask].copy()
    candidates["topic_share"] = evidence.loc[mask, client["topic"]].to_numpy(dtype=float)
    candidates["score"] = score[mask]
    order = np.argsort(-candidates["score"].to_numpy(), kind="stable")
    return candidates.iloc[order]


def pick(evidence: pd.DataFrame, client: Mapping, n: int) -> pd.DataFrame:
    """The first `n` rows of `score_client` with `pick_rank` 1..n; ValueError with fewer than `n` candidates."""
    scored = score_client(evidence, client)
    if len(scored) < n:
        raise ValueError(f"only {len(scored)} candidates after the filters, fewer than {n}")
    top = scored.iloc[:n].copy()
    top["pick_rank"] = np.arange(1, n + 1)
    return top


def followers_top(evidence: pd.DataFrame, n: int, mask=None) -> pd.DataFrame:
    """Top `n` by followers within `mask` (all rows when None); ties keep catalog order."""
    rows = evidence if mask is None else evidence.loc[np.asarray(mask, dtype=bool)]
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} creators in the mask, fewer than {n}")
    order = np.argsort(-rows["followers"].to_numpy(dtype=float), kind="stable")
    top = rows.iloc[order[:n]].copy()
    top["pick_rank"] = np.arange(1, n + 1)
    return top


def _pooled_cost(rows: pd.DataFrame) -> float:
    """Total price over total estimated Argentine reach, per 1000 (one sponsored post per creator)."""
    reach = float(rows["est_arg_reach"].sum())
    return float(rows["price_usd"].sum()) / reach * 1000.0 if reach > 0 else float("nan")


def strategy_metrics(rows: pd.DataFrame, topic: str) -> dict[str, float]:
    """The D-15 metrics of one set of creators: shares averaged, reach and price summed, cost pooled."""
    return {
        "mean_topic_share": float(rows[topic].mean()),
        "mean_argentina_audience_pct": float(rows["argentina_audience_pct"].mean()),
        "total_est_arg_reach": float(rows["est_arg_reach"].sum()),
        "mean_engagement_per_1000": float(rows["engagement_per_1000"].mean()),
        "total_price_usd": float(rows["price_usd"].sum()),
        "cost_per_1000_arg_reach": _pooled_cost(rows),
        "mean_followers": float(rows["followers"].mean()),
    }


def overlaps(picks_by_client: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """creator_id -> sorted clients that picked it, only for creators picked by two or more clients (D-12)."""
    by_creator: dict[str, set[str]] = {}
    for client, creators in picks_by_client.items():
        for creator in creators:
            by_creator.setdefault(creator, set()).add(client)
    return {creator: sorted(clients) for creator, clients in by_creator.items() if len(clients) > 1}


REGION_PROFILE_COLUMNS = (
    "client", "som_region", "n_candidates", "creators", "mean_followers", "mean_est_arg_reach",
    "mean_argentina_audience_pct", "mean_non_follower_reach_pct", "mean_engagement_per_1000",
    "mean_saves_per_1000", "mean_profile_visits_per_1000", "mean_content_tutorial_pct", "mean_price_usd",
    "cost_per_1000_arg_reach", "mean_oja_index",
)

_REGION_MEANS = (
    ("mean_followers", "followers"),
    ("mean_est_arg_reach", "est_arg_reach"),
    ("mean_argentina_audience_pct", "argentina_audience_pct"),
    ("mean_non_follower_reach_pct", "non_follower_reach_pct"),
    ("mean_engagement_per_1000", "engagement_per_1000"),
    ("mean_saves_per_1000", "saves_per_1000"),
    ("mean_profile_visits_per_1000", "profile_visits_per_1000"),
    ("mean_content_tutorial_pct", "content_tutorial_pct"),
    ("mean_price_usd", "price_usd"),
    ("mean_oja_index", "oja_index"),
)


def region_profiles(evidence: pd.DataFrame, client: Mapping) -> pd.DataFrame:
    """The client's topic candidates (topic filter only, no region filter) profiled by SOM region.

    This is the evidence for the strategy question (massive reach, discovery or educational depth):
    each map region is one strategy. Regions without candidates are omitted.
    """
    topic = evidence[client["topic"]].to_numpy(dtype=float) >= float(client["min_share"])
    rows = evidence.loc[topic]
    out = []
    for region, group in rows.groupby("som_region", sort=True):
        record = {
            "client": client["name"],
            "som_region": int(region),
            "n_candidates": int(len(group)),
            "creators": ";".join(group["name"].tolist()),
        }
        for column, source in _REGION_MEANS:
            record[column] = float(group[source].mean())
        record["cost_per_1000_arg_reach"] = _pooled_cost(group)
        out.append(record)
    return pd.DataFrame(out, columns=list(REGION_PROFILE_COLUMNS))


def _decimal_comma(value: float | int | str, spec: str = "g") -> str:
    return format(value, spec).replace(".", ",")


def describe_rule(client: Mapping) -> str:
    """The rule as one Spanish line with decimal commas (what summary.json and section 1.4 quote)."""
    parts = [f"{client['topic']} >= {_decimal_comma(float(client['min_share']))}"]
    regions = client.get("regions") or []
    if regions:
        parts.append(f"región del mapa en [{', '.join(str(r) for r in regions)}]")
    terms = " ".join(f"{_decimal_comma(float(r['weight']), '+g')}·z({r['column']})" for r in client["rank_by"])
    parts.append(f"puntaje = {terms}")
    return "; ".join(parts)
