"""Experiment kind `campaigns`: config-driven creator picks per client (Ej. 1.4).

D-10: no new model. The kind recomputes its evidence from the configs that produced runs/pca,
runs/som and runs/oja, so it equals those runs exactly and does not depend on a stale or missing
runs/ directory:
  PC1 / PC2     `fit_pca` on the same standardized design
  SOM cell      `som_evidence` on the oja config's `som_config` (bmu, grid cell, region)
  Oja index     `fit_index` on the oja config's params and seed
Each client in `clients` is a transparent rule (see tp4.analysis.campaigns): a topic-share filter,
an optional SOM-region filter and a weighted z-score. The picks are the top `n_picks`.

Config keys (besides the common ones):
  oja_config      path to the `oja` config (its som_config, data and feature_set are the single source)
  params.n_picks  creators picked per client (default 3)
  clients         list of {name, label, objective, topic, min_share, rank_by[, regions]}

Artifacts:
  evidence.csv   one row per creator: PC1/PC2, SOM cell and region, Oja index and rank, metrics
  picks.csv      the method's picks per client with the three D-13 evidence pieces
  baselines.csv  the agency's followers-only picks (D-14)

`tp4.runner` is imported only inside functions: the runner imports the experiment registry,
which imports this module (a top-level import would cycle).
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.campaigns import (
    RANKABLE_COLUMNS,
    build_evidence,
    describe_rule,
    followers_top,
    pick,
    rule_scores,
    score_client,
)
from ..analysis.pca import fit_pca, standardize
from ..config import load_config
from ..errors import ConfigError
from ..features import TOPIC_COLUMNS, load_design
from ..runs import Experiment, RunContext, resolve_project_path, write_csv
from .oja import fit_index, oja_config_from_params
from .som import load_som_config, som_evidence

EXTRA_KEYS = frozenset({"oja_config", "clients"})
PARAM_DEFAULTS: dict[str, Any] = {"n_picks": 3}
MAX_CLIENTS = 8
MAX_PICKS = 10
CLIENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")
# Names that would make a per-client summary key collide with a global one (c_picks vs n_picks, ...).
RESERVED_CLIENT_NAMES = frozenset({"n", "oja", "followers_global", "overlapping", "backend"})
REQUIRED_CLIENT_KEYS = frozenset({"name", "label", "objective", "topic", "min_share", "rank_by"})
OPTIONAL_CLIENT_KEYS = frozenset({"regions"})

PICK_COLUMNS = [
    "pick_rank", "creator_id", "name", "score", "topic_share", "pc1", "pc2", "som_region", "som_row", "som_col",
    "oja_index", "oja_rank", "followers", "argentina_audience_pct", "est_arg_reach", "engagement_per_1000",
    "price_usd", "cost_per_1000_arg_reach",
]


def _is_int(v: Any) -> bool:
    return type(v) is int


def _is_real(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def _load_oja_base(config) -> dict[str, Any]:
    """Load and validate the oja config (read only through load_config; it validates its som_config too)."""
    from .. import runner

    path = config.get("oja_config")
    if not isinstance(path, str) or not path:
        raise ConfigError(f"campaigns: oja_config must be a non-empty string path, got {path!r}")
    loaded = load_config(resolve_project_path(path))
    if loaded.get("kind") != "oja":
        raise ConfigError(f"campaigns: oja_config must be a config of kind 'oja', got kind {loaded.get('kind')!r}")
    return runner.validate_config(loaded)


def _validate_client(i: int, client: Any, seen: set[str]) -> None:
    where = f"clients[{i}]"
    if not isinstance(client, dict):
        raise ConfigError(f"campaigns: {where} must be an object, got {client!r}")
    unknown = sorted(set(client) - REQUIRED_CLIENT_KEYS - OPTIONAL_CLIENT_KEYS)
    if unknown:
        raise ConfigError(
            f"campaigns: {where} has unknown key(s) {unknown}; allowed: "
            f"{sorted(REQUIRED_CLIENT_KEYS | OPTIONAL_CLIENT_KEYS)}"
        )
    missing = sorted(REQUIRED_CLIENT_KEYS - set(client))
    if missing:
        raise ConfigError(f"campaigns: {where} is missing key(s) {missing}")
    name = client["name"]
    if not isinstance(name, str) or not CLIENT_NAME_RE.fullmatch(name):
        raise ConfigError(f"campaigns: {where}.name must match {CLIENT_NAME_RE.pattern}, got {name!r}")
    if name in RESERVED_CLIENT_NAMES:
        raise ConfigError(f"campaigns: {where}.name {name!r} is reserved ({sorted(RESERVED_CLIENT_NAMES)})")
    if name in seen:
        raise ConfigError(f"campaigns: client name {name!r} is used twice")
    seen.add(name)
    for key in ("label", "objective"):
        if not isinstance(client[key], str) or not client[key].strip():
            raise ConfigError(f"campaigns: {where}.{key} must be a non-empty string, got {client[key]!r}")
    if client["topic"] not in TOPIC_COLUMNS:
        raise ConfigError(f"campaigns: {where}.topic must be one of {list(TOPIC_COLUMNS)}, got {client['topic']!r}")
    share = client["min_share"]
    if not (_is_real(share) and 0 <= share <= 100):
        raise ConfigError(f"campaigns: {where}.min_share must be a number in [0, 100], got {share!r}")
    if "regions" in client:
        regions = client["regions"]
        if not (
            isinstance(regions, list)
            and all(_is_int(r) and r >= 1 for r in regions)
            and len(set(regions)) == len(regions)
        ):
            raise ConfigError(f"campaigns: {where}.regions must be a list of unique integers >= 1, got {regions!r}")
    rank_by = client["rank_by"]
    if not (isinstance(rank_by, list) and rank_by):
        raise ConfigError(f"campaigns: {where}.rank_by must be a non-empty list, got {rank_by!r}")
    columns: set[str] = set()
    for j, item in enumerate(rank_by):
        if not (isinstance(item, dict) and set(item) == {"column", "weight"}):
            raise ConfigError(f"campaigns: {where}.rank_by[{j}] must be an object with keys column and weight")
        if item["column"] not in RANKABLE_COLUMNS:
            raise ConfigError(
                f"campaigns: {where}.rank_by[{j}].column {item['column']!r} is not rankable; "
                f"allowed: {list(RANKABLE_COLUMNS)}"
            )
        if item["column"] in columns:
            raise ConfigError(f"campaigns: {where}.rank_by repeats column {item['column']!r}")
        columns.add(item["column"])
        if not (_is_real(item["weight"]) and item["weight"] != 0):
            raise ConfigError(f"campaigns: {where}.rank_by[{j}].weight must be a finite nonzero number, got {item['weight']!r}")


def validate(config: dict) -> None:
    _load_oja_base(config)
    n = config["params"]["n_picks"]
    if not (_is_int(n) and 1 <= n <= MAX_PICKS):
        raise ConfigError(f"campaigns: params.n_picks must be an integer in [1, {MAX_PICKS}], got {n!r}")
    clients = config.get("clients")
    if not (isinstance(clients, list) and 1 <= len(clients) <= MAX_CLIENTS):
        raise ConfigError(f"campaigns: clients must be a list of 1 to {MAX_CLIENTS} objects, got {clients!r}")
    seen: set[str] = set()
    for i, client in enumerate(clients):
        _validate_client(i, client, seen)


def _put(summary: dict[str, Any], key: str, value: Any) -> None:
    """Add a summary entry; a duplicate key (a client name shadowing a global one) is a ConfigError."""
    if key in summary:
        raise ConfigError(f"campaigns: summary key {key!r} is defined twice (check the client names)")
    summary[key] = value


def _view(rows: pd.DataFrame, client: dict, scores: np.ndarray) -> pd.DataFrame:
    """The rows of the evidence seen through a client's rule: its topic share and its rule score."""
    out = rows.copy()
    out["topic_share"] = rows[client["topic"]].to_numpy(dtype=float)
    out["score"] = scores[rows.index.to_numpy()]
    return out


def run(ctx: RunContext) -> dict:
    oja_base = _load_oja_base(ctx.config)
    som_base = load_som_config(oja_base["som_config"], who="campaigns")
    clients = list(ctx.config["clients"])
    n_picks = int(ctx.params["n_picks"])

    design = load_design(oja_base["data"], oja_base["feature_set"])
    Z = standardize(design.X)[0]
    features = list(design.X.columns)
    pca = fit_pca(Z, features)  # same oriented PCs as runs/pca (full SVD, deterministic)

    # No evidence model consumes ctx.rng: each one uses its own config's seed, so the evidence
    # equals runs/oja and runs/som exactly (asserted by tests/test_campaigns.py).
    fit = fit_index(Z, oja_config_from_params(oja_base["params"]), np.random.default_rng(oja_base["seed"]), pca.components[0])
    if fit.index is None:
        raise ConfigError("campaigns: the Oja config diverges; choose a converging eta (runs/oja_sweep)")
    som = som_evidence(som_base, Z, xp=ctx.backend.xp)
    for client in clients:
        too_high = [r for r in client.get("regions") or [] if r > som.n_regions]
        if too_high:
            raise ConfigError(
                f"campaigns: client {client['name']!r} lists region(s) {too_high} but the map has {som.n_regions} regions"
            )

    evidence = build_evidence(
        design.raw, pca.scores[:, 0], pca.scores[:, 1], fit.index, som.bmu, som.row, som.col, som.region
    )
    write_csv(ctx.artifact("evidence.csv"), evidence)

    global_top = followers_top(evidence, n_picks)
    pick_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    summary: dict[str, Any] = {}
    _put(summary, "n_creators", int(len(evidence)))
    _put(summary, "n_clients", len(clients))
    _put(summary, "n_picks", n_picks)
    _put(summary, "n_regions", int(som.n_regions))
    _put(summary, "oja_config", str(ctx.config["oja_config"]))
    _put(summary, "som_config", str(oja_base["som_config"]))
    _put(summary, "oja_eta0", float(oja_base["params"]["eta0"]))
    _put(summary, "followers_global_picks", global_top["name"].tolist())
    _put(summary, "oja_top_creator", str(evidence.loc[evidence["oja_rank"] == 1, "name"].iloc[0]))
    _put(summary, "backend_resolved", ctx.backend.name)

    for client in clients:
        scores = rule_scores(evidence, client)
        n_candidates = len(score_client(evidence, client))
        if n_candidates < n_picks:
            raise ConfigError(
                f"campaigns: client {client['name']!r} has {n_candidates} candidates after its filters, "
                f"fewer than n_picks {n_picks}"
            )
        picked = pick(evidence, client, n_picks)
        view = _view(picked, client, scores)[PICK_COLUMNS]
        view.insert(0, "label", client["label"])
        view.insert(0, "client", client["name"])
        pick_frames.append(view)

        base = _view(global_top, client, scores)[PICK_COLUMNS]
        base.insert(0, "strategy", "followers_global")
        base.insert(0, "client", client["name"])
        baseline_frames.append(base)

        c = client["name"]
        _put(summary, f"{c}_rule", describe_rule(client))
        _put(summary, f"{c}_n_candidates", int(n_candidates))
        _put(summary, f"{c}_picks", picked["name"].tolist())

    picks = pd.concat(pick_frames, ignore_index=True)
    write_csv(ctx.artifact("picks.csv"), picks)
    write_csv(ctx.artifact("baselines.csv"), pd.concat(baseline_frames, ignore_index=True))
    return summary


EXPERIMENT = Experiment(
    kind="campaigns",
    run=run,
    extra_keys=EXTRA_KEYS,
    param_defaults=PARAM_DEFAULTS,
    validate=validate,
)
