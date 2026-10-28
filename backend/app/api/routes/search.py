import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.config import SERPAPI_KEY, SEARCHAPI_KEY, SERPER_API_KEY
from app.models.price import PriceItem, ProductGroup, SearchResponse
from app.services.cache import get_cache, set_cache
from app.services.api_clients.serpapi import search_products as serpapi_search
from app.services.api_clients.searchapi import search_products as searchapi_search
from app.services.api_clients.serper import search_products as serper_search
from app.services.aggregator import (
    normalize_serpapi_results,
    normalize_searchapi_results,
    normalize_serper_results,
)
from app.services.ml import analyze
from app.services.ml.matcher import is_english
from app.services import price_log, rate_limit, source_health

logger = logging.getLogger("credisearch")

router = APIRouter(prefix="/search", tags=["search"])

FETCH_PER_SOURCE = 20


def active_sources(only: Optional[List[str]] = None):
    """
    Price sources as (name, fetch, normalize).
    A source runs only if its key is set AND it hasn't been switched off by the
    circuit breaker (out of credits / bad key -> skipped for 6 hours).
    Remove a key from the environment to retire a source for good.
    `only` restricts to the named sources (used by the daily cron to save credits).
    """
    sources = []
    if SERPAPI_KEY:
        sources.append(("serpapi", serpapi_search, normalize_serpapi_results))
    if SERPER_API_KEY:
        sources.append(("serper", serper_search, normalize_serper_results))
    if SEARCHAPI_KEY:
        sources.append(("searchapi", searchapi_search, normalize_searchapi_results))
    sources = [s for s in sources if source_health.is_available(s[0])]
    if only:
        sources = [s for s in sources if s[0] in only]
    return sources


def run_search(q: str, country: str, sources, origin: str = "user"):
    """
    Fetch every source, keep English listings, run the ML, log prices.
    Returns (enriched_items, groups, sources_ok, sources_failed, dropped_non_english).
    """
    sources_ok, sources_failed, all_items = [], [], []
    for name, fetch, normalize in sources:
        try:
            raw = fetch(q, country)
            if isinstance(raw, dict) and raw.get("error"):
                # some APIs report "out of credits" as HTTP 200 + {"error": "..."}
                raise RuntimeError(f"{name} error: {raw['error']}")
            all_items.extend(normalize(raw, limit=FETCH_PER_SOURCE))
            sources_ok.append(name)
        except Exception as exc:
            logger.warning("%s failed: %s", name, exc)
            source_health.record_failure(name, exc)
            sources_failed.append(name)

    if not sources_ok:
        raise HTTPException(
            status_code=502,
            detail="All price sources failed. Check API keys and remaining credits.",
        )

    # English listings only (drops Hindi/other-script titles before the ML sees them)
    english = [it for it in all_items if is_english(it.title)]
    dropped = len(all_items) - len(english)

    enriched, groups = analyze(english, q)
    price_log.log_search(q, groups, sources_ok, origin=origin)
    return enriched, groups, sources_ok, sources_failed, dropped


@router.get("/quota")
def search_quota(request: Request):
    """How many new searches this visitor has left today. Reading it costs nothing."""
    return rate_limit.remaining(rate_limit.visitor_id(request))


@router.get("", response_model=SearchResponse)
def search_prices(
    request: Request,
    q: str = Query(..., min_length=2, max_length=100, description="Product name"),
    country: str = Query("IN", description="Country code"),
    limit: int = Query(10, ge=1, le=40, description="Max offers returned"),
    hide_accessories: bool = Query(False, description="Drop accessories, spare parts and price outliers"),
):
    # -------- CACHE (full analysis per query; filters below are applied afterwards) --------
    cache_key = f"search:{price_log.normalize_query(q)}:{country.upper()}"
    cached = get_cache(cache_key)

    if cached:
        enriched = [PriceItem.model_validate(x) for x in cached["results"]]
        groups = [ProductGroup.model_validate(x) for x in cached["groups"]]
        sources_ok, sources_failed, dropped = cached["sources_ok"], cached["sources_failed"], cached["dropped"]
    else:
        # -------- RATE LIMIT: only fresh searches cost API credits --------
        verdict = rate_limit.check(rate_limit.visitor_id(request))
        if not verdict.allowed:
            raise HTTPException(
                status_code=429,
                detail=rate_limit.MESSAGES[verdict.reason].format(s=verdict.retry_after),
                headers={"Retry-After": str(verdict.retry_after)},
            )

        sources = active_sources()
        if not sources:
            raise HTTPException(
                status_code=503,
                detail="No price source available: keys missing, or all sources are out of credits.",
            )
        enriched, groups, sources_ok, sources_failed, dropped = run_search(q, country, sources)
        set_cache(cache_key, {
            "results": [x.model_dump() for x in enriched],
            "groups": [g.model_dump() for g in groups],
            "sources_ok": sources_ok, "sources_failed": sources_failed, "dropped": dropped,
        })

    if hide_accessories:
        enriched = [it for it in enriched if not it.is_accessory]
        groups = [g for g in groups if not g.is_accessory]
    enriched = sorted(enriched, key=lambda it: it.price)

    return SearchResponse(
        query=q,
        country=country,
        results=enriched[:limit],
        groups=groups,
        sources_ok=sources_ok,
        sources_failed=sources_failed,
        sources_disabled=source_health.disabled_sources(),
        dropped_non_english=dropped,
        history_enabled=price_log.enabled(),
        cached=bool(cached),
    )
