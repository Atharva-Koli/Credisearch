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
  tracked                   HASH   query -> {"added_at"}: what the daily job searches
  tracked_removed           HASH   query -> {"added_at", "removed_at"}: stopped, history kept
  tracked_init              STRING set once the tracked list has been seeded
  track_main                HASH   query -> history_key of its main (exact-match) product

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


MAX_TRACKED = 10


def normalize_query(q: str) -> str:
    return " ".join((q or "").lower().split())[:60]


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
    # tracked searches remember which product is "the" product for that query:
    # the best-ranked exact match (groups arrive already ranked by the ML pipeline)
    if origin in ("cron", "track"):
        main = next((g for g in candidates if g.relevance == "exact"), candidates[0] if candidates else None)
        if main is not None:
            main_key = best_by_key.get(main.history_key, main).history_key
            commands.append(["HSET", "track_main", normalize_query(query), main_key])
    _pipeline(commands)


# ------------------------------------------------------------------ tracked products

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_tracked(seed: Optional[List[str]] = None) -> List[Dict]:
    """
    The list the daily job searches. Stored in Redis so it can be edited from the UI.
    The first time it's read, it's seeded from TRACKED_PRODUCTS / the defaults.
    """
    res = _pipeline([["GET", "tracked_init"], ["HGETALL", "tracked"]])
    if res is None:
        return [{"q": q, "added_at": None} for q in (seed or [])]
    initialised, flat = res[0], res[1] or []
    if not initialised and seed:
        now = _now_iso()
        cmds = [["HSET", "tracked", normalize_query(q), json.dumps({"added_at": now})] for q in seed]
        cmds.append(["SET", "tracked_init", "1"])
        _pipeline(cmds)
        return [{"q": normalize_query(q), "added_at": now} for q in seed]
    items = [{"q": flat[i], **json.loads(flat[i + 1])} for i in range(0, len(flat), 2)]
    return sorted(items, key=lambda x: (x.get("added_at") or "", x["q"]))


def add_tracked(q: str) -> Dict:
    q = normalize_query(q)
    current = {t["q"] for t in get_tracked()}
    if q in current:
        return {"q": q, "added": False, "reason": "already tracked"}
    if len(current) >= MAX_TRACKED:
        return {"q": q, "added": False, "reason": f"limit of {MAX_TRACKED} tracked products reached"}
    _pipeline([
        ["HSET", "tracked", q, json.dumps({"added_at": _now_iso()})],
        ["HDEL", "tracked_removed", q],
        ["SET", "tracked_init", "1"],
    ])
    return {"q": q, "added": True}


def remove_tracked(q: str) -> Dict:
    """Stop tracking. The price history itself is never deleted."""
    q = normalize_query(q)
    res = _pipeline([["HGET", "tracked", q]])
    raw = (res or [None])[0]
    if not raw:
        return {"q": q, "removed": False, "reason": "not tracked"}
    info = json.loads(raw)
    _pipeline([
        ["HDEL", "tracked", q],
        ["HSET", "tracked_removed", q, json.dumps({**info, "removed_at": _now_iso()})],
        ["SET", "tracked_init", "1"],
    ])
    return {"q": q, "removed": True}


def get_timeline(seed: Optional[List[str]] = None) -> Dict:
    """
    Everything the tracker view needs in a few Redis calls: every product that is (or
    was) tracked, its main product's daily history, plus how many other products have
    history from ordinary searches.
    """
    from app.services.ml.matcher import product_key  # local import: keeps price_log light

    tracked = get_tracked(seed)
    res = _pipeline([["HGETALL", "tracked_removed"], ["HGETALL", "track_main"], ["HGETALL", "products"]])
    if res is None:
        return {"enabled": False, "tracked": [], "stopped": [], "other_products": 0, "max": MAX_TRACKED}
    rem_flat, main_flat, prod_flat = (x or [] for x in res)
    removed = [{"q": rem_flat[i], **json.loads(rem_flat[i + 1])} for i in range(0, len(rem_flat), 2)]
    mains = {main_flat[i]: main_flat[i + 1] for i in range(0, len(main_flat), 2)}
    products = {prod_flat[i]: prod_flat[i + 1] for i in range(0, len(prod_flat), 2)}

    def resolve(q: str) -> Optional[str]:
        if mains.get(q) in products:
            return mains[q]
        base = product_key(q)
        # fall back to the closest product key that starts with the query's key
        matches = sorted((k for k in products if k == base or k.startswith(base + "-")), key=len)
        return matches[0] if matches else None

    entries = [(t, "tracking") for t in tracked] + [(r, "stopped") for r in removed]
    keys = [resolve(t["q"]) for t, _ in entries]
    hist = _pipeline([["HGETALL", f"price:{k}"] for k in keys if k]) or []
    hist_iter = iter(hist)

    def build(t, status, key):
        flat = (next(hist_iter) or []) if key else []
        pts = sorted(({"date": flat[i], **json.loads(flat[i + 1])} for i in range(0, len(flat), 2)),
                     key=lambda p: p["date"])
        for p in pts:
            p.pop("best_url", None)
        summary = None
        if pts:
            low = min(pts, key=lambda p: p["min"])
            summary = {"days": len(pts), "first_date": pts[0]["date"], "last_date": pts[-1]["date"],
                       "latest_fair": pts[-1]["fair"], "latest_best": pts[-1]["min"],
                       "lowest_best": low["min"], "lowest_date": low["date"]}
        return {"q": t["q"], "status": status, "added_at": t.get("added_at"),
                "removed_at": t.get("removed_at"), "key": key,
                "name": products.get(key) if key else None, "points": pts, "summary": summary}

    built = [build(t, st, k) for (t, st), k in zip(entries, keys)]
    main_keys = {k for k in keys if k}
    return {
        "enabled": True,
        "max": MAX_TRACKED,
        "tracked": [b for b in built if b["status"] == "tracking"],
        "stopped": [b for b in built if b["status"] == "stopped"],
        "other_products": len([k for k in products if k not in main_keys]),
    }


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
