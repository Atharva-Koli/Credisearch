"""
Circuit breaker for price sources.

If a source answers with an auth / out-of-credits error (401, 402, 403, 429),
calling it again on every search just wastes time. We switch it off for
COOLDOWN_SECONDS and skip it until then.

Note: this state lives in memory, so on Vercel it resets when a new serverless
instance starts. Worst case a dead source gets one extra failed call per cold
start, which costs nothing (failed calls don't use credits).
"""

import re
import time
from typing import Dict

import httpx

COOLDOWN_SECONDS = 6 * 60 * 60  # 6 hours
QUOTA_OR_AUTH_STATUS = {401, 402, 403, 429}
QUOTA_WORDS = re.compile(
    r"credit|quota|run out|ran out|limit reached|exceeded|plan|subscription|invalid api key|unauthori[sz]ed",
    re.IGNORECASE,
)

_disabled_until: Dict[str, float] = {}
_reason: Dict[str, str] = {}


def is_available(name: str) -> bool:
    until = _disabled_until.get(name)
    if until is None:
        return True
    if time.time() >= until:
        _disabled_until.pop(name, None)
        _reason.pop(name, None)
        return True
    return False


def record_failure(name: str, exc: Exception) -> None:
    """Disable the source if the error means 'no credits' or 'bad key'."""
    reason = None
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in QUOTA_OR_AUTH_STATUS:
            reason = f"HTTP {status}"
    elif QUOTA_WORDS.search(str(exc)):
        reason = "out of credits / invalid key"
    if reason:
        _disabled_until[name] = time.time() + COOLDOWN_SECONDS
        _reason[name] = reason


def disabled_sources() -> Dict[str, str]:
    return {n: r for n, r in list(_reason.items()) if not is_available(n)}
