import httpx
from app.core.config import SERPER_API_KEY

SERPER_URL = "https://google.serper.dev/shopping"


def search_products(query: str, country: str = "IN") -> dict:
    """
    Calls Serper.dev Google Shopping and returns raw JSON response.
    Response shape: {"shopping": [{"title", "source", "link", "price": "₹65,999.00", ...}]}
    """
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY is not configured")

    payload = {
        "q": query,
        "gl": country.lower(),
        "hl": "en",
        "num": 20,
    }
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    with httpx.Client(timeout=20) as client:
        response = client.post(SERPER_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
