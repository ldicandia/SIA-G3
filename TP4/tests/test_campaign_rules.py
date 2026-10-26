import pandas as pd
import pytest

from tp4.analysis.campaigns import followers_top, pick, score_client, strategy_metrics


def test_campaign_filter_and_single_observed_criterion():
    rows = pd.DataFrame({"name": ["a", "b", "c", "d"], "topic": [70, 60, 10, 50],
                         "saves_per_1000": [4, 8, 90, 8], "followers": [100, 80, 900, 50]})
    client = {"topic": "topic", "min_share": 40, "order_by": "saves_per_1000"}
    assert score_client(rows, client)["name"].tolist() == ["b", "d", "a"]
    assert pick(rows, client, 2)["name"].tolist() == ["b", "d"]
    assert followers_top(rows, 2)["name"].tolist() == ["c", "a"]
    with pytest.raises(ValueError):
        pick(rows, client, 4)


def test_comparison_uses_original_observations():
    rows = pd.DataFrame({"topic": [60, 80], "argentina_audience_pct": [50, 90],
                         "estimated_argentina_reach": [100, 300], "non_follower_reach_pct": [20, 30],
                         "saves_per_1000": [4, 8], "content_tutorial_pct": [60, 50],
                         "profile_visits_per_1000": [10, 20], "price_usd": [40, 60], "followers": [100, 200]})
    result = strategy_metrics(rows, "topic")
    assert result["mean_topic_share"] == 70
    assert result["summed_est_arg_reach"] == 400
    assert result["total_price_usd"] == 100
