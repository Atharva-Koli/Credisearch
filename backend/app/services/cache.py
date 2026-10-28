"""
Two-level cache for search results.

1. In-memory (per serverless instance, 5 minutes): instant.
2. Redis (shared by all instances, 6 hours): repeated searches cost no API credits
   and don't count against rate limits.

Values are JSON-serialisable dicts. Redis failures are ignored (just a cache miss).
"""

import json
import time
from typing import Any, Dict, Optional, Tuple

from app.services import price_log

_CACHE: Dict[str, Tuple[float, Any]] = {}
TTL_SECONDS = 300           # in-memory
REDIS_TTL_SECONDS = 6 * 3600


def _now() -> float:
    return time.time()


def get_cache(key: str) -> Optional[Any]:
    entry = _CACHE.get(key)
    if entry:
        timestamp, data = entry
        if _now() - timestamp <= TTL_SECONDS:
            return data
        _CACHE.pop(key, None)

    if price_log.enabled():
        res = price_log._pipeline([["GET", f"cache:{key}"]])
        raw = (res or [None])[0]
        if raw:
            try:
                data = json.loads(raw)
                _CACHE[key] = (_now(), data)
                return data
            except ValueError:
                return None
    return None


def set_cache(key: str, data: Any) -> None:
    _CACHE[key] = (_now(), data)
    if price_log.enabled():
        price_log._pipeline([["SET", f"cache:{key}", json.dumps(data), "EX", REDIS_TTL_SECONDS]])
