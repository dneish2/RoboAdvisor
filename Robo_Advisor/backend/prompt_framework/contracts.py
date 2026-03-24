"""Schema contracts for symbol response payloads."""

from __future__ import annotations

from typing import Any, Dict, List

ALLOWED_EVIDENCE_TAGS = {"fact", "computed_indicator", "interpretation"}


class SchemaValidationError(ValueError):
    """Raised when response payload fails contract validation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaValidationError(message)


def validate_symbol_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a symbol-response JSON payload."""
    _require(isinstance(payload, dict), "Payload must be a JSON object")

    answer = payload.get("answer")
    claims = payload.get("claims")

    _require(isinstance(answer, str) and answer.strip(), "'answer' must be a non-empty string")
    _require(isinstance(claims, list), "'claims' must be an array")

    normalized_claims: List[Dict[str, Any]] = []
    for idx, claim in enumerate(claims):
        _require(isinstance(claim, dict), f"claims[{idx}] must be an object")

        text = claim.get("text")
        evidence_tag = claim.get("evidence_tag")
        evidence = claim.get("evidence")

        _require(isinstance(text, str) and text.strip(), f"claims[{idx}].text must be a non-empty string")
        _require(
            isinstance(evidence_tag, str) and evidence_tag in ALLOWED_EVIDENCE_TAGS,
            f"claims[{idx}].evidence_tag must be one of {sorted(ALLOWED_EVIDENCE_TAGS)}",
        )
        _require(isinstance(evidence, list) and len(evidence) > 0, f"claims[{idx}].evidence must be a non-empty array")
        for e_idx, item in enumerate(evidence):
            _require(isinstance(item, str) and item.strip(), f"claims[{idx}].evidence[{e_idx}] must be a non-empty string")

        normalized_claims.append(
            {
                "text": text.strip(),
                "evidence_tag": evidence_tag,
                "evidence": [e.strip() for e in evidence],
            }
        )

    return {"answer": answer.strip(), "claims": normalized_claims}
