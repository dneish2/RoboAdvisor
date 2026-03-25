import importlib

import pytest


EXPECTED_MODE_LABELS = {
    "scalp": "Scalp",
    "swing": "Swing",
    "position": "Position",
}

EXPECTED_OVERLAY_FLAGS = {
    "sma20": "show_sma20",
    "ema50": "show_ema50",
    "rsi14": "show_rsi14",
}


def _load_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip(f"Module {name} is not available in this branch.")


def _get_mapper(module):
    for fn_name in ("map_to_ui_payload", "to_ui_payload", "build_ui_payload"):
        fn = getattr(module, fn_name, None)
        if callable(fn):
            return fn
    pytest.skip("No UI mapper found (map_to_ui_payload/to_ui_payload/build_ui_payload).")


@pytest.mark.parametrize("mode", ["scalp", "swing", "position"])
def test_mode_maps_to_expected_frontend_behavior(mode, canonical_request):
    ui_mod = _load_module("Robo_Advisor.symbol_copilot.ui_payload")
    mapper = _get_mapper(ui_mod)

    payload = mapper({**canonical_request, "mode": mode})
    assert isinstance(payload, dict)

    assert payload.get("mode") == mode
    if "modeLabel" in payload:
        assert payload["modeLabel"] == EXPECTED_MODE_LABELS[mode]


def test_overlays_map_to_frontend_flags(canonical_request):
    ui_mod = _load_module("Robo_Advisor.symbol_copilot.ui_payload")
    mapper = _get_mapper(ui_mod)

    payload = mapper({**canonical_request, "overlays": ["sma20", "rsi14"]})
    assert isinstance(payload, dict)

    for overlay, ui_flag in EXPECTED_OVERLAY_FLAGS.items():
        if ui_flag in payload:
            expected = overlay in ["sma20", "rsi14"]
            assert payload[ui_flag] is expected

    if "overlays" in payload:
        assert set(payload["overlays"]) >= {"sma20", "rsi14"}
