"""Adapter layer for symbol-centric market data retrieval with caching."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from .cache import TTLCache

logger = logging.getLogger(__name__)


class SymbolDataAdapter:
    """Fetches quote and bar data while applying a TTL cache strategy."""

    QUOTE_TTL_SECONDS = 45
    BARS_TTL_SECONDS = 600

    def __init__(self, provider: str = "yfinance", cache: Optional[TTLCache] = None) -> None:
        self.provider = provider
        self.cache = cache or TTLCache()

    def _make_key(self, symbol: str, params: str) -> str:
        return f"{self.provider}:{symbol.upper()}:{params}"

    def get_quote(self, symbol: str) -> Dict:
        """Get latest quote snapshot for a symbol with short-lived caching."""
        cache_key = self._make_key(symbol, "quote")
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        ticker = yf.Ticker(symbol)

        try:
            history = ticker.history(period="1d", interval="1m")
            latest_price = float(history["Close"].dropna().iloc[-1]) if not history.empty else None
        except Exception as exc:
            logger.warning("Falling back to info-based quote for %s due to %s", symbol, exc)
            history = pd.DataFrame()
            latest_price = None

        info = ticker.fast_info or {}
        quote = {
            "symbol": symbol.upper(),
            "provider": self.provider,
            "last_price": latest_price if latest_price is not None else info.get("lastPrice"),
            "previous_close": info.get("previousClose"),
            "currency": info.get("currency"),
            "exchange": info.get("exchange"),
        }

        self.cache.set(cache_key, quote, self.QUOTE_TTL_SECONDS)
        return quote

    def get_bars(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get OHLCV bars for a symbol with medium-lived caching."""
        params = f"bars:{period}:{interval}"
        cache_key = self._make_key(symbol, params)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return pd.DataFrame.from_records(cached)

        ticker = yf.Ticker(symbol)
        bars = ticker.history(period=period, interval=interval)
        if bars.empty:
            return pd.DataFrame()

        bars = bars.reset_index()
        records = bars.to_dict(orient="records")
        self.cache.set(cache_key, records, self.BARS_TTL_SECONDS)
        return bars

    def get_news_stub(self, symbol: str) -> Dict:
        """Placeholder interface for future provider-backed news retrieval."""
        return {
            "symbol": symbol.upper(),
            "provider": self.provider,
            "items": [],
            "status": "not_implemented",
            "message": "News retrieval is stubbed for now; provider integration is deferred.",
        }
