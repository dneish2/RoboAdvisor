"""Versioned schemas and adapters for profile and simulation payloads.

This module introduces explicit contracts while keeping legacy request/response
shapes available for existing endpoints and components.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

import numpy as np


class GoalInputV1(TypedDict):
    """Existing goal input payload."""

    description: str
    amount: float
    time_horizon: float


class DecisionProfileV1(TypedDict, total=False):
    """Decision-profile extensions used by decision-support workflows."""

    schema_version: str
    liquidity_buffer_months: int
    max_drawdown_tolerance_pct: float
    rebalance_frequency: str
    tax_sensitivity: str


class InternalDecisionRequestV1(TypedDict, total=False):
    """Canonical internal request schema for optimization/simulation flow."""

    schema_version: str
    goals: List[GoalInputV1]
    risk_tolerance: str
    available_investment: float
    decision_profile: DecisionProfileV1


class SimulationResultV1(TypedDict):
    """Existing simulation-result schema."""

    schema_version: str
    returns: List[float]
    mean_return: float
    median_return: float
    std_dev: float
    min_return: float
    max_return: float
    percentile_5: float
    percentile_95: float


class DecisionGradeSimulationResultV1(TypedDict):
    """Decision-grade simulation schema with assumptions and quality signal."""

    schema_version: str
    base_result: SimulationResultV1
    assumptions: Dict[str, Any]
    decision_grade: Dict[str, Any]


DEFAULT_DECISION_PROFILE: DecisionProfileV1 = {
    "schema_version": "decision_profile_v1",
    "liquidity_buffer_months": 6,
    "max_drawdown_tolerance_pct": 20.0,
    "rebalance_frequency": "quarterly",
    "tax_sensitivity": "medium",
}


def adapt_legacy_input_to_internal_shape(user_profile: Dict[str, Any]) -> InternalDecisionRequestV1:
    """Adapter: old input -> new internal shape."""
    goals = user_profile.get("goals", [])

    normalized_goals: List[GoalInputV1] = []
    for goal in goals:
        normalized_goals.append(
            {
                "description": str(goal.get("description", "")).strip(),
                "amount": float(goal.get("amount", 0.0) or 0.0),
                "time_horizon": float(goal.get("time_horizon", 0.0) or 0.0),
            }
        )

    incoming_profile = user_profile.get("decision_profile") or {}
    decision_profile = {
        **DEFAULT_DECISION_PROFILE,
        **{k: v for k, v in incoming_profile.items() if v is not None},
    }

    return {
        "schema_version": "decision_profile_v1",
        "goals": normalized_goals,
        "risk_tolerance": str(user_profile.get("risk_tolerance", "Medium")),
        "available_investment": float(user_profile.get("available_investment", 0.0) or 0.0),
        "decision_profile": decision_profile,
    }


def build_simulation_result_v1(simulation_results: np.ndarray) -> SimulationResultV1:
    """Build existing simulation result schema from computed outcomes."""
    return {
        "schema_version": "simulation_result_v1",
        "returns": [float(x) for x in simulation_results.tolist()],
        "mean_return": float(np.mean(simulation_results)),
        "median_return": float(np.median(simulation_results)),
        "std_dev": float(np.std(simulation_results)),
        "min_return": float(np.min(simulation_results)),
        "max_return": float(np.max(simulation_results)),
        "percentile_5": float(np.percentile(simulation_results, 5)),
        "percentile_95": float(np.percentile(simulation_results, 95)),
    }


def build_decision_grade_simulation_result_v1(
    simulation_results: np.ndarray,
    assumptions: Dict[str, Any],
) -> DecisionGradeSimulationResultV1:
    """Build decision-grade simulation result with explicit assumptions."""
    base = build_simulation_result_v1(simulation_results)

    mean_return = base["mean_return"]
    downside = abs(base["percentile_5"])
    ratio = mean_return / downside if downside else mean_return
    if ratio >= 0.5:
        grade = "A"
    elif ratio >= 0.25:
        grade = "B"
    elif ratio >= 0.1:
        grade = "C"
    else:
        grade = "D"

    return {
        "schema_version": "decision_grade_simulation_result_v1",
        "base_result": base,
        "assumptions": assumptions,
        "decision_grade": {
            "method": "mean_to_left_tail_ratio",
            "score": float(ratio),
            "grade": grade,
        },
    }


def adapt_internal_shape_to_legacy_output_fields(
    internal_simulation: SimulationResultV1 | DecisionGradeSimulationResultV1,
) -> Dict[str, float]:
    """Adapter: new internal shape -> legacy output fields."""
    base = internal_simulation.get("base_result", internal_simulation)
    return {
        "Mean Return": float(base["mean_return"]),
        "Median Return": float(base["median_return"]),
        "Standard Deviation": float(base["std_dev"]),
        "Minimum Return": float(base["min_return"]),
        "Maximum Return": float(base["max_return"]),
        "5th Percentile": float(base["percentile_5"]),
        "95th Percentile": float(base["percentile_95"]),
    }
