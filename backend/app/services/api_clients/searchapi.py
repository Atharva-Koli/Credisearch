import httpx
from app.core.config import SEARCHAPI_KEY

SEARCHAPI_URL = "https://www.searchapi.io/api/v1/search"


def search_products(query: str, country: str = "IN") -> dict:
    """
    Calls SearchAPI Google Shopping and returns raw JSON response.
    """
    if not SEARCHAPI_KEY:
        raise RuntimeError("SEARCHAPI_KEY is not configured")

    params = {
        "engine": "google_shopping",
        "q": query,
        "country": country,
        "api_key": SEARCHAPI_KEY,
    }

    with httpx.Client(timeout=20) as client:
        response = client.get(SEARCHAPI_URL, params=params)
        response.raise_for_status()
        return response.json()
