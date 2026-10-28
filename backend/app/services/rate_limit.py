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
    verdict = Verdict(True)
    if per_min and used_min > per_min:
        verdict = Verdict(False, int(60 - (now % 60)) or 1, "minute")
    elif per_day and used_day > per_day:
        verdict = Verdict(False, seconds_to_midnight, "day")
    elif budget and used_global > budget:
        verdict = Verdict(False, seconds_to_midnight, "budget")
    if not verdict.allowed:
        # a blocked attempt costs nothing, so refund the daily counters: otherwise someone
        # hammering the button while blocked would eat the site-wide budget for everyone
        price_log._pipeline([["DECR", f"rl:d:{visitor}:{day}"], ["DECR", f"rl:g:{day}"]])
    return verdict


def remaining(visitor: str, now: Optional[float] = None) -> dict:
    """Read-only view of what's left today for this visitor (doesn't use anything up)."""
    per_min = _int_env("RATE_LIMIT_PER_MINUTE", 5)
    per_day = _int_env("RATE_LIMIT_PER_DAY", 30)
    budget = _int_env("DAILY_SEARCH_BUDGET", 40)
    now = time.time() if now is None else now
    minute = int(now // 60)
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    res = price_log._pipeline([
        ["GET", f"rl:m:{visitor}:{minute}"], ["GET", f"rl:d:{visitor}:{day}"], ["GET", f"rl:g:{day}"],
    ]) if price_log.enabled() else None
    if not res:
        return {"enabled": False}
    used_min, used_day, used_global = (int(x or 0) for x in res)

    def left(limit, used):
        return None if not limit else max(0, limit - used)

    you, site = left(per_day, used_day), left(budget, used_global)
    today = min(x for x in (you, site) if x is not None) if (you is not None or site is not None) else None
    return {
        "enabled": True,
        "left_today": today,                 # what this visitor can actually still do today
        "you": {"left": you, "limit": per_day or None},
        "site": {"left": site, "limit": budget or None},
        "this_minute": {"left": left(per_min, used_min), "limit": per_min or None},
        "resets_in": int(86400 - (now % 86400)),
    }


MESSAGES = {
    "minute": "You're searching quite fast. Please wait {s} seconds and try again.",
    "day": "You've reached today's search limit. Searches you've already made still load instantly.",
    "budget": "Credisearch has used today's search budget (it runs on free API tiers). "
              "Searches made earlier today still work; new ones reset at 5:30 AM IST.",
}
