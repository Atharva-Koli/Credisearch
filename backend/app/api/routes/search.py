from fastapi import APIRouter, Query
from app.models.price import SearchResponse

from app.services.cache import get_cache, set_cache

from app.services.api_clients.serpapi import search_products as serpapi_search
from app.services.api_clients.searchapi import search_products as searchapi_search

from app.services.aggregator import (
    normalize_serpapi_results,
    normalize_searchapi_results,
    merge_price_results,
)

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
def search_prices(
    q: str = Query(..., description="Product name"),
    country: str = Query("IN", description="Country code"),
    limit: int = Query(5, ge=1, le=20),
):
    # -------- CACHE CHECK --------
    cache_key = f"{q}:{country}:{limit}"
    cached_response = get_cache(cache_key)
    if cached_response:
        return cached_response

    # -------- FETCH RAW DATA --------
    serp_raw = serpapi_search(q, country)
    search_raw = searchapi_search(q, country)

    # -------- NORMALIZE --------
    # fetch more internally, limit only after merge
    serp_items = normalize_serpapi_results(serp_raw, limit=20)
    search_items = normalize_searchapi_results(search_raw, limit=20)

    # -------- MERGE & SORT --------
    merged_results = merge_price_results(
        serp_items,
        search_items,
        limit=limit,
    )

    # -------- RESPONSE --------
    response = SearchResponse(
        query=q,
        country=country,
        results=merged_results,
    )

    # -------- CACHE STORE --------
    set_cache(cache_key, response)

    return response
