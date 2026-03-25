import importlib

import pytest


def _load_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip(f"Module {name} is not available in this branch.")


def _get_cache_cls(module):
    for cls_name in ("DataAdapterCache", "TTLCache", "DataCache"):
        cls = getattr(module, cls_name, None)
        if cls is not None:
            return cls
    pytest.skip("No cache class found (DataAdapterCache/TTLCache/DataCache).")


def test_cache_hit_miss_and_ttl_behavior(monkeypatch):
    adapter_mod = _load_module("Robo_Advisor.symbol_copilot.data_adapter")
    cache_cls = _get_cache_cls(adapter_mod)

    # Deterministic virtual clock.
    now = {"t": 1_700_000_000.0}

    def fake_time():
        return now["t"]

    if not hasattr(adapter_mod, "time"):
        pytest.skip("data_adapter module does not expose time for deterministic TTL testing.")

    monkeypatch.setattr(adapter_mod.time, "time", fake_time)

    cache = cache_cls(ttl_seconds=5)

    # Initial miss.
    assert cache.get("AAPL") is None

    cache.set("AAPL", {"price": 189.42})

    # Hit within TTL.
    assert cache.get("AAPL") == {"price": 189.42}

    # TTL expiry should return miss.
    now["t"] += 6
    assert cache.get("AAPL") is None
