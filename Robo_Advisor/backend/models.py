# backend/models.py

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionProfile:
    """Canonical representation of user decision intent for simulation defaults."""

    objective_preset: str
    risk_stance: str
    success_definition: str
    thesis_summary: str


@dataclass(frozen=True)
class SimulationAssumptions:
    """Deterministic defaults derived from a DecisionProfile."""

    num_simulations: int
    horizon_multiplier: float
    return_bias_multiplier: float
    tail_event_floor: float


_OBJECTIVE_BASELINES = {
    "capital_preservation": SimulationAssumptions(
        num_simulations=5000,
        horizon_multiplier=0.9,
        return_bias_multiplier=0.9,
        tail_event_floor=-0.08,
    ),
    "balanced_growth": SimulationAssumptions(
        num_simulations=10000,
        horizon_multiplier=1.0,
        return_bias_multiplier=1.0,
        tail_event_floor=-0.12,
    ),
    "aggressive_growth": SimulationAssumptions(
        num_simulations=20000,
        horizon_multiplier=1.1,
        return_bias_multiplier=1.15,
        tail_event_floor=-0.2,
    ),
    "income": SimulationAssumptions(
        num_simulations=8000,
        horizon_multiplier=1.0,
        return_bias_multiplier=0.95,
        tail_event_floor=-0.1,
    ),
}

_RISK_ADJUSTMENTS = {
    "low": SimulationAssumptions(2000, 0.95, 0.9, -0.06),
    "medium": SimulationAssumptions(0, 1.0, 1.0, -0.1),
    "high": SimulationAssumptions(5000, 1.1, 1.1, -0.18),
}


class UserProfile:
    def __init__(self, goals, risk_tolerance, available_investment, decision_profile: Optional[Dict[str, Any]] = None):
        """
        Initialize the UserProfile with goals, risk tolerance, available investment, and optional decision profile.

        :param goals: List of dictionaries, each containing 'description', 'amount', and 'time_horizon'
        :param risk_tolerance: String indicating risk level ('Low', 'Medium', 'High')
        :param available_investment: Float indicating the amount available for investment
        :param decision_profile: Optional dictionary representation of DecisionProfile
        """
        self.goals = goals
        self.risk_tolerance = risk_tolerance
        self.available_investment = available_investment
        self.decision_profile = decision_profile

    def save_to_csv(self, filepath):
        try:
            goals_rounded = [{**goal, "time_horizon": round(goal["time_horizon"], 1)} for goal in self.goals]

            decision_profile = normalize_decision_profile(
                self.decision_profile,
                legacy_risk_tolerance=self.risk_tolerance,
                legacy_goals=goals_rounded,
            )

            df = pd.DataFrame(
                [
                    {
                        "goals": json.dumps(goals_rounded),
                        "risk_tolerance": self.risk_tolerance,
                        "available_investment": self.available_investment,
                        "decision_profile": json.dumps(asdict(decision_profile)),
                        "thesis_summary": decision_profile.thesis_summary,
                    }
                ]
            )

            write_header = not os.path.exists(filepath)

            df.to_csv(filepath, mode="a", header=write_header, index=False)
            logger.info(
                f"User profile saved successfully to {filepath}. Headers {'written' if write_header else 'not written'}."
            )
        except Exception as e:
            logger.error(f"Error saving user profile to CSV: {e}")

    @staticmethod
    def load_from_csv(filepath):
        """
        Load user profiles from a CSV file.

        :param filepath: Path to the CSV file
        :return: Pandas DataFrame containing user profiles
        """
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                expected_columns = ["goals", "risk_tolerance", "available_investment"]
                if all(col in df.columns for col in expected_columns):
                    logger.info(f"Loaded user profiles from {filepath} with expected columns.")
                    return df
                # Handle cases where headers might be missing
                logger.warning(f"Expected columns missing in {filepath}. Assigning default column names.")
                df = pd.read_csv(filepath, header=None, names=["goals", "risk_tolerance", "available_investment"])
                return df
            except pd.errors.ParserError as e:
                logger.error(f"ParserError while loading user profiles: {e}")
                # Attempt to read without headers
                logger.info(f"Attempting to read {filepath} without headers.")
                df = pd.read_csv(filepath, header=None, names=["goals", "risk_tolerance", "available_investment"])
                return df
            except Exception as e:
                logger.error(f"Error loading user profiles from CSV: {e}")
                return pd.DataFrame()
        logger.warning(f"File {filepath} does not exist. Returning empty DataFrame.")
        return pd.DataFrame()


def normalize_decision_profile(
    decision_profile: Optional[Dict[str, Any]],
    legacy_risk_tolerance: str,
    legacy_goals: Optional[list] = None,
) -> DecisionProfile:
    """Return explicit decision profile, or derive one from legacy fields."""
    if isinstance(decision_profile, str):
        try:
            decision_profile = json.loads(decision_profile)
        except json.JSONDecodeError:
            decision_profile = None

    if isinstance(decision_profile, dict):
        objective_preset = str(decision_profile.get("objective_preset") or "balanced_growth").strip().lower()
        risk_stance = str(decision_profile.get("risk_stance") or legacy_risk_tolerance or "Medium").strip().lower()
        success_definition = str(decision_profile.get("success_definition") or "Meet stated goals with controlled drawdown.").strip()
        thesis_summary = str(decision_profile.get("thesis_summary") or "").strip()
        return DecisionProfile(
            objective_preset=objective_preset,
            risk_stance=risk_stance,
            success_definition=success_definition,
            thesis_summary=thesis_summary,
        )

    return derive_decision_profile_from_legacy(legacy_risk_tolerance=legacy_risk_tolerance, goals=legacy_goals or [])


def derive_decision_profile_from_legacy(legacy_risk_tolerance: str, goals: list) -> DecisionProfile:
    """Fallback path: produce a deterministic profile from legacy goal/risk fields."""
    risk = (legacy_risk_tolerance or "Medium").strip().lower()
    objective = {"low": "capital_preservation", "medium": "balanced_growth", "high": "aggressive_growth"}.get(
        risk,
        "balanced_growth",
    )

    max_horizon = 0
    target_amount = 0.0
    for goal in goals:
        if not isinstance(goal, dict):
            continue
        target_amount += float(goal.get("amount", 0) or 0)
        max_horizon = max(max_horizon, float(goal.get("time_horizon", 0) or 0))

    success_definition = f"Pursue ${target_amount:,.0f} across {max_horizon:.1f} years with {risk} risk controls."
    thesis_summary = "Derived from legacy goals/risk fields; provide a thesis summary to improve explainability metadata."

    return DecisionProfile(
        objective_preset=objective,
        risk_stance=risk,
        success_definition=success_definition,
        thesis_summary=thesis_summary,
    )


def decision_profile_to_simulation_assumptions(profile: DecisionProfile) -> SimulationAssumptions:
    """Deterministic mapping from DecisionProfile to simulation defaults."""
    objective = profile.objective_preset.strip().lower()
    risk = profile.risk_stance.strip().lower()

    objective_defaults = _OBJECTIVE_BASELINES.get(objective, _OBJECTIVE_BASELINES["balanced_growth"])
    risk_delta = _RISK_ADJUSTMENTS.get(risk, _RISK_ADJUSTMENTS["medium"])

    return SimulationAssumptions(
        num_simulations=max(1000, objective_defaults.num_simulations + risk_delta.num_simulations),
        horizon_multiplier=objective_defaults.horizon_multiplier * risk_delta.horizon_multiplier,
        return_bias_multiplier=objective_defaults.return_bias_multiplier * risk_delta.return_bias_multiplier,
        tail_event_floor=min(objective_defaults.tail_event_floor, risk_delta.tail_event_floor),
    )
