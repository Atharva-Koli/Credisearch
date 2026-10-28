"""
Price history + the daily logging job.

  GET /history                  every product seen so far (key -> name)
  GET /history/{key}            daily price points for one product
  GET /history/events/recent    the latest search events (the "event log")
  GET /cron/log-prices          run by Vercel Cron once a day: searches the
                                tracked products and logs their prices

  GET    /tracker               tracked + previously tracked products with their timelines
  POST   /tracker               start tracking a product   (needs X-Admin-Token)
  DELETE /tracker?q=...         stop tracking; history is kept (needs X-Admin-Token)
"""

import hmac
import os
import time

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.services import price_log

router = APIRouter(tags=["history"])

# Products searched automatically every day so history keeps growing.
# These only SEED the list the first time: after that the list lives in Redis and is
# edited from the UI (Price tracker tab) or the /tracker API.
DEFAULT_TRACKED = [
    "iphone 17",
    "iphone 16",
    "oneplus 15r",
    "oppo reno 15 pro mini 5g",
    "samsung galaxy s25",
]
# The cron prefers Serper (2,500 credits) so it doesn't eat SerpAPI's 250/month.
CRON_SOURCES = ["serper"]
CRON_FALLBACK_SOURCES = ["serpapi"]


def tracked_products():
    raw = os.getenv("TRACKED_PRODUCTS")
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    return DEFAULT_TRACKED


@router.get("/history")
def list_history():
    if not price_log.enabled():
        return {"enabled": False, "products": {}}
    return {"enabled": True, "products": price_log.list_products()}


@router.get("/history/events/recent")
def recent_events(limit: int = Query(50, ge=1, le=500)):
    return {"enabled": price_log.enabled(), "events": price_log.recent_events(limit)}


@router.get("/history/{key}")
def product_history(key: str):
    if not price_log.enabled():
        raise HTTPException(status_code=503, detail="Price history is not configured (no Redis env vars).")
    data = price_log.get_history(key)
    pts = data["points"]
    if pts:
        lowest = min(pts, key=lambda p: p["min"])
        data["summary"] = {
            "days_tracked": len(pts),
            "first_date": pts[0]["date"],
            "lowest_min": lowest["min"],
            "lowest_date": lowest["date"],
            "latest_fair": pts[-1]["fair"],
        }
    return data


# ------------------------------------------------------------------ tracker

class TrackRequest(BaseModel):
    q: str = Field(..., min_length=2, max_length=60, description="Product to track, e.g. 'iphone 17'")


def _require_admin(token: str) -> None:
    """Editing the tracked list costs API credits every day, so it's admin-only."""
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=403, detail="Editing is disabled: set ADMIN_TOKEN in Vercel.")
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Wrong admin key.")


def _cron_sources():
    from app.api.routes.search import active_sources  # avoid import cycle
    return active_sources(only=CRON_SOURCES) or active_sources(only=CRON_FALLBACK_SOURCES)


@router.get("/tracker")
def tracker():
    data = price_log.get_timeline(seed=tracked_products())
    data["can_edit"] = bool(os.getenv("ADMIN_TOKEN"))
    return data


@router.post("/tracker")
def track_product(body: TrackRequest, x_admin_token: str = Header(None)):
    _require_admin(x_admin_token)
    if not price_log.enabled():
        raise HTTPException(status_code=503, detail="Price history is not configured (no Redis env vars).")
    result = price_log.add_tracked(body.q)
    if not result["added"]:
        raise HTTPException(status_code=409, detail=result["reason"])
    # record today's price straight away so the product appears on the timeline now
    from app.api.routes.search import run_search
    sources = _cron_sources()
    if sources:
        try:
            run_search(result["q"], "IN", sources, origin="track")
            result["logged_now"] = True
        except Exception as exc:
            result["logged_now"] = False
            result["log_error"] = str(exc)[:200]
    return result


@router.delete("/tracker")
def untrack_product(q: str = Query(..., min_length=2), x_admin_token: str = Header(None)):
    _require_admin(x_admin_token)
    result = price_log.remove_tracked(q)
    if not result["removed"]:
        raise HTTPException(status_code=404, detail=result["reason"])
    return result


@router.get("/cron/log-prices", include_in_schema=False)
def cron_log_prices(authorization: str = Header(None)):
    # Vercel Cron sends "Authorization: Bearer <CRON_SECRET>" when CRON_SECRET is set
    secret = os.getenv("CRON_SECRET")
    if secret and authorization != f"Bearer {secret}":
        raise HTTPException(status_code=401, detail="unauthorized")
    if not price_log.enabled():
        return {"ok": False, "reason": "price logger not configured"}

    from app.api.routes.search import run_search  # avoid import cycle

    sources = _cron_sources()
    if not sources:
        return {"ok": False, "reason": "no source available for cron"}

    done, failed = [], []
    start = time.time()
    # the editable list in Redis (seeded from TRACKED_PRODUCTS / defaults the first time)
    for q in [t["q"] for t in price_log.get_tracked(seed=tracked_products())]:
        if time.time() - start > 50:  # stay inside the function time limit
            break
        try:
            run_search(q, "IN", sources, origin="cron")
            done.append(q)
        except Exception as exc:
            failed.append({"q": q, "error": str(exc)[:200]})
    return {"ok": True, "source": sources[0][0], "logged": done, "failed": failed}
