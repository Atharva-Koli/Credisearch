from typing import List
from app.models.price import PriceItem

USD_TO_INR = 83.0  # temporary fixed rate


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

            price = item.get("extracted_price")

            if price is None:
                price_raw = item.get("price")
                if price_raw:
                    price = float(
                        price_raw.replace("₹", "")
                        .replace(",", "")
                        .strip()
                    )

            if not title or not price or not link:
                continue

            results.append(
                PriceItem(
                    source="searchapi",
                    title=title,
                    price=float(price),
                    currency="INR",
                    url=link,
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
