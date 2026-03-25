"""Lightweight SQLite-backed TTL cache utilities for market data."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional


class TTLCache:
    """A minimal SQLite-based TTL cache for small market-data payloads."""

    def __init__(self, db_path: str = "Robo_Advisor/storage/market_data_cache.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    expires_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache(expires_at)"
            )
            conn.commit()

    def get(self, cache_key: str) -> Optional[Any]:
        now = int(time.time())
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

            if row is None:
                return None

            payload, expires_at = row
            if expires_at <= now:
                conn.execute("DELETE FROM cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
                return None

            return json.loads(payload)

    def set(self, cache_key: str, value: Any, ttl_seconds: int) -> None:
        expires_at = int(time.time()) + max(1, int(ttl_seconds))
        payload = json.dumps(value)

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO cache (cache_key, payload, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload=excluded.payload,
                    expires_at=excluded.expires_at
                """,
                (cache_key, payload, expires_at),
            )
            conn.commit()

    def purge_expired(self) -> int:
        now = int(time.time())
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
            conn.commit()
            return cursor.rowcount
