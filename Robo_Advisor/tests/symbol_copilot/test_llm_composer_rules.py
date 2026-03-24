import importlib

import pytest


def _load_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip(f"Module {name} is not available in this branch.")


def _compose(module, request_payload):
    for fn_name in ("compose_response", "build_response", "generate_response"):
        fn = getattr(module, fn_name, None)
        if callable(fn):
            return fn(request_payload)
    pytest.skip("No composer function found (compose_response/build_response/generate_response).")


def test_contrarian_and_followups_are_always_populated(canonical_request):
    composer = _load_module("Robo_Advisor.symbol_copilot.llm_composer")

    # Exercise the same request with slight deterministic variants.
    modes = ["scalp", "swing", "position"]
    for mode in modes:
        payload = {**canonical_request, "mode": mode}
        response = _compose(composer, payload)

        assert isinstance(response.get("contrarian"), str)
        assert response["contrarian"].strip(), "contrarian must be non-empty"

        followups = response.get("followups")
        assert isinstance(followups, list), "followups must be a list"
        assert followups, "followups must contain at least one item"
        assert all(isinstance(item, str) and item.strip() for item in followups)
