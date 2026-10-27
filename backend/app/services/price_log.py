"""
Price event logger + price history, stored in Upstash Redis (free tier) through
its REST API, so no extra Python packages are needed.

What gets stored
----------------
  events                    LIST   newest-first log of every search (capped at 5,000)
  products                  HASH   history_key -> display name of every product seen
  price:<history_key>       HASH   'YYYY-MM-DD' -> {"fair", "min", "max", "offers", "best_url"}
                                   (one point per product per day; the latest search that
                                   day wins)

Data comes from two places:
  1. every user search (free: no extra API credits)
  2. a daily Vercel cron that searches a short list of tracked products, so the
     history keeps growing even on days nobody searches

If the Redis env vars are missing, every function here quietly does nothing.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger("credisearch")

EVENTS_CAP = 5000

# Vercel's Upstash integration names them KV_REST_API_*; Upstash itself uses UPSTASH_REDIS_REST_*
REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL") or os.getenv("KV_REST_API_URL")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN") or os.getenv("KV_REST_API_TOKEN")


def enabled() -> bool:
    return bool(REDIS_URL and REDIS_TOKEN)


def _pipeline(commands: List[list]) -> Optional[list]:
    """Send several Redis commands in one HTTP request."""
    if not enabled() or not commands:
        return None
    try:
        with httpx.Client(timeout=4) as client:
            r = client.post(
                f"{REDIS_URL.rstrip('/')}/pipeline",
                headers={"Authorization": f"Bearer {REDIS_TOKEN}"},
                json=commands,
            )
            r.raise_for_status()
            return [item.get("result") for item in r.json()]
    except Exception as exc:  # logging must never break a search
        logger.warning("price log failed: %s", exc)
        return None


def log_search(query: str, groups, sources_ok: List[str], origin: str = "user") -> None:
    """Record one search: an event + one daily price point per relevant product."""
    if not enabled():
        return
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%d")

    candidates = [
        g for g in groups
        if not g.is_accessory and not g.is_refurbished and g.relevance != "other" and g.history_key
    ]
    # two groups can map to the same product key: keep the one with more evidence
    best_by_key = {}
    for g in candidates:
        if g.history_key not in best_by_key or g.offers > best_by_key[g.history_key].offers:
            best_by_key[g.history_key] = g
    tracked = sorted(best_by_key.values(), key=lambda g: -g.offers)

    event = {
        "ts": now.isoformat(timespec="seconds"),
        "origin": origin,                     # "user" or "cron"
        "q": query,
        "sources": sources_ok,
        "products": [
            {"key": g.history_key, "fair": g.median_price, "min": g.best_offer_price, "offers": g.offers}
            for g in tracked[:5]
        ],
    }
    commands: List[list] = [
        ["LPUSH", "events", json.dumps(event)],
        ["LTRIM", "events", 0, EVENTS_CAP - 1],
    ]
    for g in tracked[:5]:
        point = {
            # "min" is the best TRUSTED price (suspicious offers excluded), not the raw minimum
            "fair": g.median_price, "min": g.best_offer_price, "max": g.max_price,
            "offers": g.offers, "best_url": g.best_offer_url,
        }
        commands.append(["HSET", "products", g.history_key, g.name])
        commands.append(["HSET", f"price:{g.history_key}", day, json.dumps(point)])
    _pipeline(commands)


def get_history(key: str) -> Dict:
    res = _pipeline([["HGETALL", f"price:{key}"], ["HGET", "products", key]])
    if not res:
        return {"key": key, "name": None, "points": []}
    flat, name = res[0] or [], res[1]
    points = [
        {"date": flat[i], **json.loads(flat[i + 1])}
        for i in range(0, len(flat), 2)
    ]
    points.sort(key=lambda p: p["date"])
    return {"key": key, "name": name, "points": points}


def list_products() -> Dict[str, str]:
    res = _pipeline([["HGETALL", "products"]])
    flat = (res or [[]])[0] or []
    return {flat[i]: flat[i + 1] for i in range(0, len(flat), 2)}


def recent_events(limit: int = 50) -> List[dict]:
    res = _pipeline([["LRANGE", "events", 0, max(0, limit - 1)]])
    return [json.loads(e) for e in ((res or [[]])[0] or [])]
