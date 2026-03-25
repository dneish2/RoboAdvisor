"""Feature-flag helpers for staged rollout of versioned payload contracts."""

from __future__ import annotations

import os
from typing import Dict

FEATURE_FLAGS: Dict[str, bool] = {
    "decision_profile_v1": False,
    "simulation_assumptions_v1": False,
    "recommendation_loop_v1": False,
}


def _to_bool(raw_value: str | None, default: bool) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def is_enabled(flag_name: str) -> bool:
    """Return whether a feature flag is enabled via environment variable override."""
    if flag_name not in FEATURE_FLAGS:
        raise KeyError(f"Unknown feature flag: {flag_name}")

    env_name = f"ROBO_{flag_name.upper()}"
    return _to_bool(os.getenv(env_name), FEATURE_FLAGS[flag_name])


def all_flags() -> Dict[str, bool]:
    """Return all effective flag values for observability/debugging."""
    return {name: is_enabled(name) for name in FEATURE_FLAGS}
