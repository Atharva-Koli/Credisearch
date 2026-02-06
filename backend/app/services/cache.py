import time
from typing import Dict, Tuple, Any

# key -> (timestamp, data)
_CACHE: Dict[str, Tuple[float, Any]] = {}
TTL_SECONDS = 300  # 5 minutes


def _now() -> float:
    return time.time()


def get_cache(key: str):
    entry = _CACHE.get(key)
    if not entry:
        return None

    timestamp, data = entry
    if _now() - timestamp > TTL_SECONDS:
        _CACHE.pop(key, None)
        return None

    return data


def set_cache(key: str, data):
    _CACHE[key] = (_now(), data)
