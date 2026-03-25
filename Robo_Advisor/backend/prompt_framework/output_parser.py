"""Utilities to parse and repair model JSON outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict


def _extract_json_candidate(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return "{}"

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _repair_json_string(candidate: str) -> str:
    repaired = candidate.strip()
    repaired = re.sub(r"```(?:json)?", "", repaired, flags=re.IGNORECASE).replace("```", "").strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def parse_or_repair_json(raw_text: str) -> Dict[str, Any]:
    """Attempt to parse JSON output and apply lightweight repairs."""
    candidate = _extract_json_candidate(raw_text)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = _repair_json_string(candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Last resort: single-quote normalization for common model slips.
            normalized = repaired.replace("'", '"')
            return json.loads(normalized)
