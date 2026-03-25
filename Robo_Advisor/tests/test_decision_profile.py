import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

from backend.models import (
    DecisionProfile,
    decision_profile_to_simulation_assumptions,
    derive_decision_profile_from_legacy,
    normalize_decision_profile,
)


def test_decision_profile_mapping_is_deterministic():
    profile = DecisionProfile(
        objective_preset="aggressive_growth",
        risk_stance="high",
        success_definition="Outperform benchmark.",
        thesis_summary="Concentrate in secular growth leaders.",
    )

    first = decision_profile_to_simulation_assumptions(profile)
    second = decision_profile_to_simulation_assumptions(profile)

    assert first == second
    assert first.num_simulations == 25000


def test_legacy_fallback_derives_profile_when_missing_new_profile():
    goals = [{"description": "Retirement", "amount": 500000, "time_horizon": 15}]

    derived = derive_decision_profile_from_legacy("Low", goals)

    assert derived.objective_preset == "capital_preservation"
    assert "500,000" in derived.success_definition


def test_normalize_decision_profile_accepts_json_string():
    payload = json.dumps(
        {
            "objective_preset": "income",
            "risk_stance": "medium",
            "success_definition": "Fund annual cashflow.",
            "thesis_summary": "Dividend quality focus.",
        }
    )

    profile = normalize_decision_profile(payload, legacy_risk_tolerance="Medium", legacy_goals=[])

    assert profile.objective_preset == "income"
    assert profile.thesis_summary == "Dividend quality focus."
