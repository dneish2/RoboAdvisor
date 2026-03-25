"""Deterministic recommendation ranking engine.

This module intentionally avoids LLM usage for:
- numeric calculations
- ranking logic
- simulation logic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ActionTemplate:
    """Static template that defines deterministic action behavior."""

    action_id: str
    title: str
    thesis: str
    assumptions: List[str]
    complexity_penalty: float
    base_confidence: float


ACTION_TEMPLATES: List[ActionTemplate] = [
    ActionTemplate(
        action_id="de_risk_equity",
        title="Reduce equity beta by 5%",
        thesis="Volatility and drawdown are elevated, so reducing equity beta can improve downside resilience.",
        assumptions=[
            "Portfolio has flexible allocation limits.",
            "Current market regime is risk-off or uncertain.",
        ],
        complexity_penalty=0.14,
        base_confidence=0.72,
    ),
    ActionTemplate(
        action_id="rebalance_concentration",
        title="Trim top holding concentration",
        thesis="Concentration risk is above target, so trimming a large position can reduce idiosyncratic risk.",
        assumptions=[
            "Top holding can be rebalanced without major tax friction.",
            "Replacement assets are liquid and diversified.",
        ],
        complexity_penalty=0.10,
        base_confidence=0.77,
    ),
    ActionTemplate(
        action_id="add_quality_tilt",
        title="Add quality tilt (+4%)",
        thesis="Momentum is constructive and volatility is contained, supporting a measured quality tilt.",
        assumptions=[
            "Momentum signal is persistent over the near-term horizon.",
            "Risk budget allows incremental cyclical exposure.",
        ],
        complexity_penalty=0.08,
        base_confidence=0.69,
    ),
    ActionTemplate(
        action_id="raise_cash_buffer",
        title="Raise cash buffer (+3%)",
        thesis="Cash buffer is low versus uncertainty, so increasing liquidity can improve optionality.",
        assumptions=[
            "User values drawdown control over full upside capture.",
            "Short-term rates remain acceptable for idle cash.",
        ],
        complexity_penalty=0.05,
        base_confidence=0.66,
    ),
]


THRESHOLDS: Dict[str, float] = {
    "high_volatility_30d": 0.32,
    "deep_drawdown_90d": -0.15,
    "strong_momentum_30d": 0.08,
    "high_concentration_top_holding": 0.24,
    "low_cash_buffer": 0.05,
}


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _signal_strength(indicators: Dict[str, Any]) -> float:
    """Deterministically aggregate regime strength from thresholds."""
    vol = _to_float(indicators.get("volatility_30d"), THRESHOLDS["high_volatility_30d"])
    dd = _to_float(indicators.get("drawdown_90d"), THRESHOLDS["deep_drawdown_90d"])
    mom = _to_float(indicators.get("momentum_30d"), 0.0)
    concentration = _to_float(indicators.get("concentration_top_holding"), 0.0)
    cash = _to_float(indicators.get("cash_buffer"), 0.08)

    stressed = float(vol > THRESHOLDS["high_volatility_30d"]) + float(dd < THRESHOLDS["deep_drawdown_90d"])
    constructive = float(mom > THRESHOLDS["strong_momentum_30d"])
    fragility = float(concentration > THRESHOLDS["high_concentration_top_holding"]) + float(
        cash < THRESHOLDS["low_cash_buffer"]
    )
    return _clip((stressed + constructive + fragility) / 5.0, 0.0, 1.0)


def _impact_delta(template: ActionTemplate, indicators: Dict[str, Any]) -> float:
    """Compute deterministic impact delta for each action."""
    vol = _to_float(indicators.get("volatility_30d"), THRESHOLDS["high_volatility_30d"])
    dd = _to_float(indicators.get("drawdown_90d"), THRESHOLDS["deep_drawdown_90d"])
    mom = _to_float(indicators.get("momentum_30d"), 0.0)
    concentration = _to_float(indicators.get("concentration_top_holding"), 0.0)
    cash = _to_float(indicators.get("cash_buffer"), 0.08)

    if template.action_id == "de_risk_equity":
        score = (vol - THRESHOLDS["high_volatility_30d"]) * 40 + (THRESHOLDS["deep_drawdown_90d"] - dd) * 35
    elif template.action_id == "rebalance_concentration":
        score = (concentration - THRESHOLDS["high_concentration_top_holding"]) * 120
    elif template.action_id == "add_quality_tilt":
        score = (mom - THRESHOLDS["strong_momentum_30d"]) * 90 - max(0.0, vol - THRESHOLDS["high_volatility_30d"]) * 35
    else:  # raise_cash_buffer
        score = (THRESHOLDS["low_cash_buffer"] - cash) * 140 + max(0.0, vol - THRESHOLDS["high_volatility_30d"]) * 15

    return _clip(score / 100.0, -0.15, 0.25)


def _confidence(template: ActionTemplate, indicators: Dict[str, Any]) -> float:
    available_fields = ["volatility_30d", "drawdown_90d", "momentum_30d", "concentration_top_holding", "cash_buffer"]
    coverage = sum(1 for field in available_fields if indicators.get(field) is not None) / len(available_fields)
    strength = _signal_strength(indicators)
    return _clip(template.base_confidence * 0.65 + coverage * 0.20 + strength * 0.15, 0.0, 0.99)


def _impact_transition(impact_delta: float) -> Dict[str, float]:
    before_pct = 50.0
    after_pct = _clip(before_pct + impact_delta * 100.0, 30.0, 80.0)
    return {"impact_before_pct": round(before_pct, 1), "impact_after_pct": round(after_pct, 1)}


def rank_actions(indicators: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    """Return top-k deterministic actions sorted by final score."""
    scored: List[Dict[str, Any]] = []
    for template in ACTION_TEMPLATES:
        impact_delta = _impact_delta(template, indicators)
        confidence = _confidence(template, indicators)
        final_score = impact_delta * confidence - template.complexity_penalty
        transition = _impact_transition(impact_delta)

        scored.append(
            {
                "action_id": template.action_id,
                "title": template.title,
                "thesis": template.thesis,
                "assumptions": template.assumptions,
                "impact_delta": round(impact_delta, 4),
                "confidence_score": round(confidence, 4),
                "complexity_penalty": template.complexity_penalty,
                "score": round(final_score, 4),
                **transition,
            }
        )

    scored.sort(key=lambda item: (-item["score"], item["action_id"]))
    return scored[: max(1, top_k)]


def build_recommendation_blocks(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """Build deterministic UI blocks for What happened / Why / What next."""
    ranked_actions = rank_actions(indicators=indicators, top_k=3)

    vol = _to_float(indicators.get("volatility_30d"), THRESHOLDS["high_volatility_30d"])
    dd = _to_float(indicators.get("drawdown_90d"), THRESHOLDS["deep_drawdown_90d"])
    mom = _to_float(indicators.get("momentum_30d"), 0.0)

    happened = (
        f"30d momentum={mom:.1%}, 30d volatility={vol:.1%}, "
        f"90d drawdown={dd:.1%}. Thresholds are deterministic and fixed."
    )
    top_action = ranked_actions[0]
    why = {
        "thesis": top_action["thesis"],
        "assumptions": top_action["assumptions"],
    }
    return {
        "what_happened": happened,
        "why": why,
        "what_next": ranked_actions,
        "ranking_method": "score = impact_delta * confidence_score - complexity_penalty",
        "thresholds": THRESHOLDS,
    }
