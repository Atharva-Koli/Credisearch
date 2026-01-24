import httpx
from app.core.config import SERPAPI_KEY

SERPAPI_URL = "https://serpapi.com/search.json"


def search_products(query: str, country: str = "IN") -> dict:
    """
    Calls SerpAPI Google Shopping and returns raw JSON response.
    """
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY is not configured")

    params = {
        "engine": "google_shopping",
        "q": query,
        "gl": country,
        "hl": "en",
        "api_key": SERPAPI_KEY,
    }

    with httpx.Client(timeout=20) as client:
        response = client.get(SERPAPI_URL, params=params)
        response.raise_for_status()
        return response.json()
