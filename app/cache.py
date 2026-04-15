"""
cache.py
--------
Simple in-memory cache to avoid recomputing expensive graph analytics
on every API call.  Uses a TTL-based invalidation strategy.

For production scale-out, swap `_store` with Redis using the same interface.
"""

import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

_store: dict[str, dict] = {}   # { key: { "value": ..., "expires_at": float } }

DEFAULT_TTL = 3600  # seconds (1 hour)


def set_cache(key: str, value: Any, ttl: int = DEFAULT_TTL) -> None:
    """Store a value in the cache with a TTL expiry."""
    _store[key] = {
        "value": value,
        "expires_at": time.time() + ttl,
    }
    logger.debug(f"Cache SET: '{key}' (TTL={ttl}s)")


def get_cache(key: str) -> Any:
    """Retrieve a cached value or None if missing / expired."""
    entry = _store.get(key)
    if entry is None:
        return None
    if time.time() > entry["expires_at"]:
        del _store[key]
        logger.debug(f"Cache EXPIRED: '{key}'")
        return None
    logger.debug(f"Cache HIT: '{key}'")
    return entry["value"]


def invalidate(key: str) -> None:
    """Remove a specific key from the cache."""
    _store.pop(key, None)
    logger.debug(f"Cache INVALIDATED: '{key}'")


def clear_all() -> None:
    """Wipe the entire cache."""
    _store.clear()
    logger.info("Cache CLEARED")


def cache_info() -> dict:
    """Return diagnostic info about cached keys."""
    now = time.time()
    return {
        "total_keys": len(_store),
        "valid_keys": sum(1 for e in _store.values() if e["expires_at"] > now),
        "expired_keys": sum(1 for e in _store.values() if e["expires_at"] <= now),
        "keys": list(_store.keys()),
    }
