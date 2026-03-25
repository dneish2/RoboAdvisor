"""Prompt templates for symbol analysis flows."""

from typing import Any
import json


def build_symbol_analysis_prompt(query: str, context: Any) -> str:
    """Build user prompt for symbol analysis.

    Args:
        query: User question.
        context: Precomputed context payload (dict/list/string).
    """
    serialized_context = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False, default=str)

    return (
        "You are given market context and a user query. "
        "Return a JSON response that follows the required schema.\n\n"
        f"User query:\n{query}\n\n"
        f"Context:\n{serialized_context}\n\n"
        "Requirements:\n"
        "- Include concise answer text in 'answer'.\n"
        "- Add one or more claims in 'claims'.\n"
        "- Each claim must include evidence_tag in {fact, computed_indicator, interpretation}.\n"
        "- Use explicit snippets/fields from context in each claim's evidence array."
    )
