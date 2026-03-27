import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.data_fetcher import DataFetcher
from backend.portfolio_optimizer import PortfolioOptimizer


def test_investment_thesis_source_file_has_required_columns():
    """Main functional requirement: app can source investable universe from CSV."""
    fetcher = DataFetcher()
    thesis = fetcher.load_investment_thesis("data/investment_thesis.csv")

    assert not thesis.empty
    assert {"Ticker", "Theme", "RiskLevel"}.issubset(set(thesis.columns))
    assert "AAPL" in thesis["Ticker"].values


def test_optimizer_risk_filter_respects_max_holdings():
    """Main functional requirement: risk filtering + holding cap controls portfolio universe."""
    optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)
    optimizer.risk_tolerance = "Low"
    optimizer.max_holdings = 2
    optimizer.investment_thesis = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC", "DDD"],
            "Theme": ["X", "X", "Y", "Y"],
            "RiskLevel": ["Low", "Low", "Medium", "High"],
        }
    )

    selected = optimizer.select_assets_based_on_risk()
    assert selected == ["AAA", "BBB"]


def test_optimizer_emits_canonical_decision_grade_payload():
    """Main functional requirement: simulation output contract remains stable."""
    optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)
    optimizer.time_horizon_years = 5
    optimizer.risk_tolerance = "Medium"

    payload = optimizer._build_simulation_payload(np.array([0.04, -0.01, 0.02, 0.03], dtype=float))

    assert payload["schema_version"] == "decision_grade_simulation_result_v1"
    assert "base_result" in payload
    assert payload["assumptions"]["sampling_method"] == "historical_bootstrap"


def test_investment_thesis_loader_normalizes_and_deduplicates(tmp_path):
    thesis_file = tmp_path / "thesis.csv"
    thesis_file.write_text(
        "Ticker,Theme,RiskLevel\n"
        "aapl,Tech,medium\n"
        " AAPL ,Tech,MEDIUM\n"
        "msft,Tech,low\n",
        encoding="utf-8",
    )

    fetcher = DataFetcher()
    thesis = fetcher.load_investment_thesis(str(thesis_file))

    assert thesis["Ticker"].tolist() == ["AAPL", "MSFT"]
    assert thesis["RiskLevel"].tolist() == ["Medium", "Low"]
