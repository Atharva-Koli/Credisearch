"""
Price history + the daily logging job.

  GET /history                  every product seen so far (key -> name)
  GET /history/{key}            daily price points for one product
  GET /history/events/recent    the latest search events (the "event log")
  GET /cron/log-prices          run by Vercel Cron once a day: searches the
                                tracked products and logs their prices
"""

import os
import time

from fastapi import APIRouter, Header, HTTPException, Query

from app.services import price_log

router = APIRouter(tags=["history"])

# Products searched automatically every day so history keeps growing.
# Override with TRACKED_PRODUCTS="iphone 16 128gb, pixel 9 ..." in Vercel.
DEFAULT_TRACKED = [
    "iphone 15 128gb",
    "iphone 16 128gb",
    "samsung galaxy s24 256gb",
    "oneplus 13r 256gb",
    "sony wh-1000xm5",
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


@router.get("/cron/log-prices", include_in_schema=False)
def cron_log_prices(authorization: str = Header(None)):
    # Vercel Cron sends "Authorization: Bearer <CRON_SECRET>" when CRON_SECRET is set
    secret = os.getenv("CRON_SECRET")
    if secret and authorization != f"Bearer {secret}":
        raise HTTPException(status_code=401, detail="unauthorized")
    if not price_log.enabled():
        return {"ok": False, "reason": "price logger not configured"}

    from app.api.routes.search import active_sources, run_search  # avoid import cycle

    sources = active_sources(only=CRON_SOURCES) or active_sources(only=CRON_FALLBACK_SOURCES)
    if not sources:
        return {"ok": False, "reason": "no source available for cron"}

    done, failed = [], []
    start = time.time()
    for q in tracked_products():
        if time.time() - start > 50:  # stay inside the function time limit
            break
        try:
            run_search(q, "IN", sources, origin="cron")
            done.append(q)
        except Exception as exc:
            failed.append({"q": q, "error": str(exc)[:200]})
    return {"ok": True, "source": sources[0][0], "logged": done, "failed": failed}
