"""
Rate limiting for fresh (uncached) searches, which cost up to 3 API credits each.

Three fixed-window counters in Redis, shared by every serverless instance:

  rl:m:<visitor>:<minute>   per-visitor per-minute   (RATE_LIMIT_PER_MINUTE, default 5)
  rl:d:<visitor>:<day>      per-visitor per-day      (RATE_LIMIT_PER_DAY, default 30)
  rl:g:<day>                whole-site daily budget  (DAILY_SEARCH_BUDGET, default 40)

Cached searches never reach this, so repeating a search is always free.
A visitor is identified by a hash of their IP address (the raw IP is never stored).
If Redis is unavailable the limiter fails open: searches keep working.
"""

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Optional

from app.services import price_log


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, default)))
    except ValueError:
        return default


@dataclass
class Verdict:
    allowed: bool
    retry_after: int = 0          # seconds until the blocking window resets
    reason: Optional[str] = None  # "minute" | "day" | "budget"


def visitor_id(request) -> str:
    """Hash of the client IP. On Vercel, x-real-ip / x-forwarded-for are set by the edge."""
    ip = (
        request.headers.get("x-real-ip")
        or (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        or (request.client.host if request.client else "")
        or "unknown"
    )
    salt = os.getenv("CRON_SECRET", "credisearch")  # keeps hashes from being reversible by lookup
    return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()[:16]


def check(visitor: str, now: Optional[float] = None) -> Verdict:
    per_min = _int_env("RATE_LIMIT_PER_MINUTE", 5)
    per_day = _int_env("RATE_LIMIT_PER_DAY", 30)
    budget = _int_env("DAILY_SEARCH_BUDGET", 40)

    if not price_log.enabled():
        return Verdict(True)

    now = time.time() if now is None else now
    minute = int(now // 60)
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    res = price_log._pipeline([
        ["INCR", f"rl:m:{visitor}:{minute}"], ["EXPIRE", f"rl:m:{visitor}:{minute}", 120],
        ["INCR", f"rl:d:{visitor}:{day}"],    ["EXPIRE", f"rl:d:{visitor}:{day}", 90000],
        ["INCR", f"rl:g:{day}"],              ["EXPIRE", f"rl:g:{day}", 90000],
    ])
    if not res:  # Redis down -> fail open
        return Verdict(True)
    used_min, used_day, used_global = int(res[0]), int(res[2]), int(res[4])

    seconds_to_midnight = int(86400 - (now % 86400)) or 1
    if per_min and used_min > per_min:
        return Verdict(False, int(60 - (now % 60)) or 1, "minute")
    if per_day and used_day > per_day:
        return Verdict(False, seconds_to_midnight, "day")
    if budget and used_global > budget:
        return Verdict(False, seconds_to_midnight, "budget")
    return Verdict(True)


MESSAGES = {
    "minute": "You're searching quite fast. Please wait {s} seconds and try again.",
    "day": "You've reached today's search limit. Searches you've already made still load instantly.",
    "budget": "Credisearch has used today's search budget (it runs on free API tiers). "
              "Searches made earlier today still work; new ones reset at 5:30 AM IST.",
}
