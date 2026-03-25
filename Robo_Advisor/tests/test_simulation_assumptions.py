import pytest
import os
import sys

# Keep imports consistent with repo's module layout.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.portfolio_optimizer import (
    PortfolioOptimizer,
    SimulationAssumptions,
    map_legacy_ui_to_simulation_assumptions,
)


def test_legacy_mapper_preserves_defaults_for_existing_flow():
    user_profile = {
        "goals": [{"description": "Retirement", "amount": 500000, "time_horizon": 20}],
        "risk_tolerance": "Medium",
        "available_investment": 10000,
    }

    assumptions = map_legacy_ui_to_simulation_assumptions(user_profile)

    assert assumptions.expected_return == pytest.approx(0.08)
    assert assumptions.volatility == pytest.approx(0.16)
    assert assumptions.inflation_rate == pytest.approx(0.02)
    assert assumptions.contribution_frequency_per_year == 12
    assert assumptions.scenario_toggles["market_stress"] is False


def test_guardrails_raise_on_hard_bound_violations():
    optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)
    optimizer.time_horizon_years = 5
    optimizer.simulation_assumptions = SimulationAssumptions(
        expected_return=0.90,  # invalid
        volatility=0.20,
        inflation_rate=0.02,
        correlation=0.2,
        periodic_contribution=100,
        periodic_withdrawal=0,
        contribution_frequency_per_year=12,
        withdrawal_frequency_per_year=12,
        rebalance_cadence="quarterly",
        scenario_toggles={},
    )

    with pytest.raises(ValueError):
        optimizer.validate_simulation_assumptions()


def test_guardrails_warn_but_do_not_crash_for_low_confidence():
    optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)
    optimizer.time_horizon_years = 20
    optimizer.simulation_assumptions = SimulationAssumptions(
        expected_return=0.16,
        volatility=0.40,
        inflation_rate=0.10,
        correlation=0.90,
        periodic_contribution=100,
        periodic_withdrawal=500,
        contribution_frequency_per_year=12,
        withdrawal_frequency_per_year=12,
        rebalance_cadence="quarterly",
        scenario_toggles={},
    )

    optimizer.validate_simulation_assumptions()
    assert len(optimizer.simulation_warnings) >= 3
