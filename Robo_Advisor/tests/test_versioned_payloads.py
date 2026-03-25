import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.versioned_payloads import (
    adapt_internal_shape_to_legacy_output_fields,
    adapt_legacy_input_to_internal_shape,
    build_decision_grade_simulation_result_v1,
    build_simulation_result_v1,
)


def test_adapt_legacy_input_to_internal_shape_builds_decision_profile():
    legacy = {
        "goals": [{"description": "Retire", "amount": 500000, "time_horizon": 20}],
        "risk_tolerance": "Medium",
        "available_investment": 20000,
    }

    internal = adapt_legacy_input_to_internal_shape(legacy)

    assert internal["schema_version"] == "decision_profile_v1"
    assert internal["decision_profile"]["schema_version"] == "decision_profile_v1"
    assert internal["goals"][0]["description"] == "Retire"


def test_simulation_schema_and_legacy_projection_are_compatible():
    simulation_results = np.array([0.1, -0.03, 0.05, 0.02], dtype=float)

    v1 = build_simulation_result_v1(simulation_results)
    decision_grade = build_decision_grade_simulation_result_v1(
        simulation_results,
        assumptions={"sampling_method": "historical_bootstrap"},
    )

    legacy_from_v1 = adapt_internal_shape_to_legacy_output_fields(v1)
    legacy_from_decision = adapt_internal_shape_to_legacy_output_fields(decision_grade)

    assert v1["schema_version"] == "simulation_result_v1"
    assert decision_grade["schema_version"] == "decision_grade_simulation_result_v1"
    assert legacy_from_v1["Mean Return"] == legacy_from_decision["Mean Return"]
    assert "5th Percentile" in legacy_from_decision
