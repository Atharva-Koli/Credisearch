import logging

from fastapi import APIRouter, HTTPException, Query

from app.models.price import SearchResponse
from app.services.cache import get_cache, set_cache
from app.services.api_clients.serpapi import search_products as serpapi_search
from app.services.api_clients.searchapi import search_products as searchapi_search
from app.services.aggregator import (
    normalize_serpapi_results,
    normalize_searchapi_results,
    merge_price_results,
)
from app.services.ml import analyze

logger = logging.getLogger("credisearch")

router = APIRouter(prefix="/search", tags=["search"])

FETCH_PER_SOURCE = 20


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

    # -------- FETCH (each source independently; one failing must not kill the search) --------
    sources_ok, sources_failed = [], []
    serp_items, search_items = [], []

    try:
        serp_items = normalize_serpapi_results(serpapi_search(q, country), limit=FETCH_PER_SOURCE)
        sources_ok.append("serpapi")
    except Exception as exc:
        logger.warning("serpapi failed: %s", exc)
        sources_failed.append("serpapi")

    try:
        search_items = normalize_searchapi_results(searchapi_search(q, country), limit=FETCH_PER_SOURCE)
        sources_ok.append("searchapi")
    except Exception as exc:
        logger.warning("searchapi failed: %s", exc)
        sources_failed.append("searchapi")

    if not sources_ok:
        raise HTTPException(
            status_code=502,
            detail="All price sources failed. Check SERPAPI_KEY / SEARCHAPI_KEY and API quotas.",
        )

    # -------- MERGE (all offers; ML needs the full picture before we trim) --------
    all_items = merge_price_results(serp_items, search_items, limit=FETCH_PER_SOURCE * 2)

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
    )

    # -------- CACHE STORE --------
    set_cache(cache_key, response)
    return response
