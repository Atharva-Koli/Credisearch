import re
from typing import List
from app.models.price import PriceItem

USD_TO_INR = 83.0  # temporary fixed rate


def _image(item: dict):
    """Product thumbnail: SerpAPI/SearchAPI call it 'thumbnail', Serper 'imageUrl'."""
    img = item.get("thumbnail") or item.get("imageUrl") or item.get("image")
    if isinstance(img, str) and (img.startswith("https://") or img.startswith("data:image/")):
        return img
    return None


def normalize_serpapi_results(data: dict, limit: int = 5):
    """
    Converts raw SerpAPI Google Shopping response into PriceItem objects.
    Normalizes all prices to INR.
    """
    results = []
    shopping_results = data.get("shopping_results", [])

    for item in shopping_results:
        try:
            title = item.get("title")
            link = item.get("product_link")

            price_raw = item.get("price")
            extracted_price = item.get("extracted_price")

            if not title or not link:
                continue

            price_inr = None

            # Case 1: Price string exists → check currency symbol
            if price_raw:
                if "₹" in price_raw:
                    price_inr = float(
                        price_raw.replace("₹", "")
                        .replace(",", "")
                        .strip()
                    )
                elif "$" in price_raw:
                    price_usd = float(
                        price_raw.replace("$", "")
                        .replace(",", "")
                        .strip()
                    )
                    price_inr = price_usd * USD_TO_INR

            # Case 2: Fallback to extracted_price
            elif extracted_price:
                # Heuristic: small values are USD, large are INR
                if extracted_price < 2000:
                    price_inr = extracted_price * USD_TO_INR
                else:
                    price_inr = extracted_price

            if not price_inr:
                continue

            results.append(
                PriceItem(
                    source="serpapi",
                    title=title,
                    price=float(price_inr),
                    currency="INR",
                    url=link,
                    image=_image(item),
                )
            )

            if len(results) >= limit:
                break

        except Exception:
            continue

    return results



# Additional normalization function for SearchAPI results:


def normalize_searchapi_results(data: dict, limit: int = 5):
    """
    Converts raw SearchAPI Google Shopping response into PriceItem objects.
    """
    results = []

    shopping_results = data.get("shopping_results", [])

    for item in shopping_results:
        try:
            title = item.get("title")
            link = item.get("link") or item.get("product_link")

            price_raw = item.get("price") or ""
            price = item.get("extracted_price")

            if price is None and price_raw:
                price = float(
                    re.sub(r"[^\d.]", "", price_raw.replace(",", "")) or 0
                )

            if not title or not price or not link:
                continue

            # Currency safety net: same USD->INR rule as the SerpAPI path.
            # Any other foreign currency is dropped rather than mislabelled.
            if "$" in price_raw:
                price = float(price) * USD_TO_INR
            elif price_raw and "₹" not in price_raw and re.search(r"[€£¥]", price_raw):
                continue

            results.append(
                PriceItem(
                    source="searchapi",
                    title=title,
                    price=float(price),
                    currency="INR",
                    url=link,
                    image=_image(item),
                )
            )

            if len(results) >= limit:
                break

        except Exception:
            continue

    return results


def _price_to_inr(price_raw: str):
    """'₹65,999.00' -> 65999.0 ; '$309.00' -> converted ; other currencies -> None."""
    if not price_raw:
        return None
    number = re.sub(r"[^\d.]", "", price_raw.replace(",", ""))
    if not number:
        return None
    value = float(number)
    if "₹" in price_raw or "rs" in price_raw.lower():
        return value
    if "$" in price_raw:
        return value * USD_TO_INR
    if re.search(r"[€£¥]", price_raw):
        return None
    return value  # no symbol: assume local currency (we request gl=in)


def normalize_serper_results(data: dict, limit: int = 5):
    """
    Converts raw Serper.dev Google Shopping response into PriceItem objects.
    """
    results = []
    for item in data.get("shopping", []):
        try:
            title = item.get("title")
            link = item.get("link")
            price = _price_to_inr(item.get("price") or "")
            if not title or not link or not price:
                continue
            seller = item.get("source")
            results.append(
                PriceItem(
                    source=f"serper · {seller}" if seller else "serper",
                    title=title,
                    price=float(price),
                    currency="INR",
                    url=link,
                    image=_image(item),
                )
            )
            if len(results) >= limit:
                break
        except Exception:
            continue
    return results


#sorting and merging function to get best prices
def merge_price_results(
    serpapi_results: list,
    searchapi_results: list,
    limit: int = 5,
):
    """
    Merge and sort price results from multiple sources.
    """
    combined = serpapi_results + searchapi_results

    # Sort by price (ascending)
    combined.sort(key=lambda x: x.price)

    return combined[:limit]
