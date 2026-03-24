import importlib

import pytest


REQUIRED_FIELDS = {
    "symbol": str,
    "mode": str,
    "summary": str,
    "signals": list,
    "contrarian": str,
    "followups": list,
    "confidence": (int, float),
    "asof": str,
}


def _load_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip(f"Module {name} is not available in this branch.")


def _get_response_from_module(module, request_payload):
    for fn_name in ("compose_response", "build_response", "generate_response"):
        fn = getattr(module, fn_name, None)
        if callable(fn):
            return fn(request_payload)
    pytest.skip("No response composer function found (compose_response/build_response/generate_response).")


def test_response_contract_has_required_fields_and_types(canonical_request):
    composer = _load_module("Robo_Advisor.symbol_copilot.llm_composer")
    response = _get_response_from_module(composer, canonical_request)

    assert isinstance(response, dict), "Composer must return a dict payload."

    missing = [field for field in REQUIRED_FIELDS if field not in response]
    assert not missing, f"Missing required fields: {missing}"

    for field, expected_type in REQUIRED_FIELDS.items():
        assert isinstance(response[field], expected_type), (
            f"Field '{field}' must be {expected_type}, got {type(response[field])}"
        )

    assert response["symbol"], "symbol must be non-empty"
    assert response["summary"].strip(), "summary must be non-empty"
    assert 0 <= float(response["confidence"]) <= 1, "confidence expected in [0,1]"
