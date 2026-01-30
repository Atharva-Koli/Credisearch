from fastapi import APIRouter, Query
from app.models.price import SearchResponse
from app.services.api_clients.serpapi import search_products
from app.services.aggregator import normalize_serpapi_results

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
def search_prices(
    q: str = Query(..., description="Product name"),
    country: str = Query("IN", description="Country code"),
    limit: int = Query(5, ge=1, le=20),
):
    raw_data = search_products(q, country)
    results = normalize_serpapi_results(raw_data, limit=limit)

    return SearchResponse(
        query=q,
        country=country,
        results=results,
    )
