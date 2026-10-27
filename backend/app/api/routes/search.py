import logging

from fastapi import APIRouter, HTTPException, Query

from app.core.config import SERPAPI_KEY, SEARCHAPI_KEY, SERPER_API_KEY
from app.models.price import SearchResponse
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
from app.services import source_health

logger = logging.getLogger("credisearch")

router = APIRouter(prefix="/search", tags=["search"])

FETCH_PER_SOURCE = 20


def active_sources():
    """
    Price sources as (name, fetch, normalize).
    A source runs only if its key is set AND it hasn't been switched off by the
    circuit breaker (out of credits / bad key -> skipped for 6 hours).
    Remove a key from the environment to retire a source for good.
    """
    sources = []
    if SERPAPI_KEY:
        sources.append(("serpapi", serpapi_search, normalize_serpapi_results))
    if SERPER_API_KEY:
        sources.append(("serper", serper_search, normalize_serper_results))
    if SEARCHAPI_KEY:
        sources.append(("searchapi", searchapi_search, normalize_searchapi_results))
    return [s for s in sources if source_health.is_available(s[0])]


@router.get("", response_model=SearchResponse)
def search_prices(
    q: str = Query(..., min_length=2, description="Product name"),
    country: str = Query("IN", description="Country code"),
    limit: int = Query(10, ge=1, le=40, description="Max offers returned"),
    hide_accessories: bool = Query(False, description="Drop cases, chargers, etc."),
):
    # -------- CACHE CHECK --------
    cache_key = f"{q.lower().strip()}:{country}:{limit}:{hide_accessories}"
    cached_response = get_cache(cache_key)
    if cached_response:
        return cached_response

    sources = active_sources()
    if not sources:
        raise HTTPException(
            status_code=503,
            detail="No price source available: keys missing, or all sources are out of credits.",
        )

    # -------- FETCH (each source independently; one failing must not kill the search) --------
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

    # -------- ML: product matching + deal scoring --------
    enriched, groups = analyze(all_items, q)

    if hide_accessories:
        enriched = [it for it in enriched if not it.is_accessory]
        groups = [g for g in groups if not g.is_accessory]

    enriched.sort(key=lambda it: it.price)

    response = SearchResponse(
        query=q,
        country=country,
        results=enriched[:limit],
        groups=groups,
        sources_ok=sources_ok,
        sources_failed=sources_failed,
        sources_disabled=source_health.disabled_sources(),
    )

    # -------- CACHE STORE --------
    set_cache(cache_key, response)
    return response
