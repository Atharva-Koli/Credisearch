from fastapi import APIRouter, Query
from app.models.price import SearchResponse

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
    # Fetch raw data from both sources
    serp_raw = serpapi_search(q, country)
    search_raw = searchapi_search(q, country)

    # Normalize results
    serp_items = normalize_serpapi_results(serp_raw, limit=limit)
    search_items = normalize_searchapi_results(search_raw, limit=limit)

    # Merge & sort by price
    merged_results = merge_price_results(
        serp_items,
        search_items,
        limit=limit,
    )

    return SearchResponse(
        query=q,
        country=country,
        results=merged_results,
    )
