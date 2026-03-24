import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def bars_fixture(fixture_dir):
    return json.loads((fixture_dir / "bars.json").read_text())


@pytest.fixture(scope="session")
def quotes_fixture(fixture_dir):
    return json.loads((fixture_dir / "quotes.json").read_text())


@pytest.fixture(scope="session")
def news_fixture(fixture_dir):
    return json.loads((fixture_dir / "news.json").read_text())


@pytest.fixture(scope="session")
def canonical_request(bars_fixture, quotes_fixture, news_fixture):
    return {
        "symbol": "AAPL",
        "bars": bars_fixture,
        "quote": quotes_fixture,
        "news": news_fixture,
        "mode": "swing",
        "overlays": ["sma20", "rsi14"],
    }
