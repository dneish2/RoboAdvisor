import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.recommendation_engine import build_recommendation_blocks, rank_actions


def test_rank_actions_is_deterministic_and_top3():
    indicators = {
        "volatility_30d": 0.38,
        "drawdown_90d": -0.21,
        "momentum_30d": 0.04,
        "concentration_top_holding": 0.31,
        "cash_buffer": 0.03,
    }

    first = rank_actions(indicators, top_k=3)
    second = rank_actions(indicators, top_k=3)

    assert first == second
    assert len(first) == 3
    assert all("impact_before_pct" in action and "impact_after_pct" in action for action in first)


def test_build_recommendation_blocks_schema():
    blocks = build_recommendation_blocks(
        {
            "volatility_30d": 0.25,
            "drawdown_90d": -0.08,
            "momentum_30d": 0.12,
            "concentration_top_holding": 0.18,
            "cash_buffer": 0.07,
        }
    )

    assert isinstance(blocks["what_happened"], str) and blocks["what_happened"]
    assert "thesis" in blocks["why"]
    assert isinstance(blocks["why"].get("assumptions"), list)
    assert isinstance(blocks["what_next"], list) and len(blocks["what_next"]) == 3
    assert "impact_delta" in blocks["what_next"][0]
