import json

import numpy as np
import pandas as pd
import pytest

from tp4.cli import main
from tp4.config import load_config
from tp4.errors import ConfigError
from tp4.runner import validate_config
from tp4.runs import PROJECT_ROOT


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("campaigns")
    assert main(["run", "configs/campaigns.json", "--out-root", str(out)]) == 0
    return out / "campaigns"


def test_required_comparisons_and_evidence(run_dir):
    evidence = pd.read_csv(run_dir / "evidence.csv")
    picks = pd.read_csv(run_dir / "picks.csv")
    comparisons = pd.read_csv(run_dir / "comparison.csv")
    assert len(evidence) == 50
    assert len(picks) == 12
    assert set(comparisons["strategy"]) == {"assumption", "followers_global", "followers_in_topic"}
    assert {"som_bmu", "pc1", "pc2", "oja_index"}.issubset(evidence.columns)
    config = load_config(PROJECT_ROOT / "configs/campaigns.json")
    for client in config["clients"]:
        rows = picks.loc[picks["client"] == client["name"]]
        assert (rows[client["topic"]] >= client["min_share"]).all()
        expected = evidence.loc[evidence[client["topic"]] >= client["min_share"]]
        expected = expected.sort_values(client["order_by"], ascending=False, kind="stable").head(3)
        assert rows["creator_id"].tolist() == expected["creator_id"].tolist()
    assert (run_dir / "comparison.png").stat().st_size > 0


def test_evidence_agrees_with_separate_models(run_dir, tmp_path):
    evidence = pd.read_csv(run_dir / "evidence.csv").set_index("creator_id")
    for kind in ("oja", "som"):
        cfg = load_config(PROJECT_ROOT / f"configs/{kind}.json")
        cfg["plots"] = False
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps(cfg))
        assert main(["run", str(path), "--out-root", str(tmp_path)]) == 0
    oja = pd.read_csv(tmp_path / "oja/ranking.csv").set_index("creator_id")
    som = pd.read_csv(tmp_path / "som/bmus.csv").set_index("creator_id")
    np.testing.assert_allclose(evidence["oja_index"], oja.loc[evidence.index, "oja_index"])
    np.testing.assert_array_equal(evidence["som_bmu"], som.loc[evidence.index, "bmu"])


def test_campaign_rejects_unsupported_ranking_keys():
    cfg = load_config(PROJECT_ROOT / "configs/campaigns.json")
    cfg["clients"][0]["rank_by"] = [{"column": "followers", "weight": 1}]
    with pytest.raises(ConfigError):
        validate_config(cfg)
