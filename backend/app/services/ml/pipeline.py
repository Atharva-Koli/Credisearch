from collections import defaultdict
from statistics import median
from typing import List, Tuple

from app.models.price import PriceItem, ProductGroup
from app.services.ml.matcher import (
    accessory_reason, canonical_key, cluster_titles, display_name, extract_attributes,
    is_used, normalize, product_key,
)
from app.services.ml.deal_scorer import score_group

# A product group priced below this fraction of the main product's price is
# almost certainly an accessory or part that slipped past the word lists.
PRICE_OUTLIER_RATIO = 0.20
# ...and one priced above 5x the anchor is a mis-priced / wrong-currency listing.
PRICE_OUTLIER_HIGH_RATIO = 5.0

# A product must share at least half of the searched words to count as relevant.
MIN_QUERY_OVERLAP = 0.5


def _group_name(titles: List[str]) -> str:
    """Most representative title (median length), with colour words stripped."""
    by_len = sorted(titles, key=len)
    return display_name(by_len[len(by_len) // 2])


def history_key(titles: List[str]) -> str:
    """
    Stable id for a product across searches ('apple-iphone-15-128gb'): the most
    common canonical key among the group's titles (ties -> the shortest).
    """
    keys = [product_key(t) for t in titles]
    return min(set(keys), key=lambda k: (-keys.count(k), len(k)))


def _mismatch(query: str, group_name: str) -> int:
    """0 = exactly what was searched; 1-2 = a variant / other storage; 3+ = a different model."""
    q_attrs = extract_attributes(query)
    q_words = set(normalize(query).split())
    a = extract_attributes(group_name)
    score = 0
    # off-topic: shares fewer than half of the searched words ("alfa hockey stick" -> Skechers shoes)
    q_terms = set(canonical_key(query).split("-")) - {""}
    g_terms = set(canonical_key(group_name).split("-"))
    if q_terms and len(q_terms & g_terms) / len(q_terms) < MIN_QUERY_OVERLAP:
        score += 3
    if q_attrs["model_codes"] and a["model_codes"] and not (q_attrs["model_codes"] & a["model_codes"]):
        score += 3
    if q_attrs["model_nums"] and a["model_nums"] and not (q_attrs["model_nums"] & a["model_nums"]):
        score += 3
    score += len(a["variants"] - q_words)            # extra variant (searched "15", got "15 Pro")
    score += len(q_attrs["variants"] - a["variants"])  # missing variant (searched "15 Pro", got "15")
    if q_attrs["storage"] and a["storage"] and a["storage"] != q_attrs["storage"]:
        score += 1
    return score


def _relevance(mismatch: int) -> str:
    return "exact" if mismatch == 0 else ("variant" if mismatch < 3 else "other")


def analyze(items: List[PriceItem], query: str) -> Tuple[List[PriceItem], List[ProductGroup]]:
    """
    Enrich offers with product-group ids, accessory/part flags and deal verdicts.
    Returns (enriched_items, groups) with groups ranked best-first.
    """
    if not items:
        return [], []

    labels = cluster_titles([it.title for it in items])

    members = defaultdict(list)
    for idx, lab in enumerate(labels):
        members[lab].append(idx)

    enriched: List[PriceItem] = [None] * len(items)  # type: ignore
    groups: List[ProductGroup] = []

    for lab, idxs in members.items():
        group_items = [items[i] for i in idxs]
        verdicts = score_group([it.price for it in group_items])

        # majority vote: is this group an accessory / part / add-on?
        reasons = [accessory_reason(it.title, query) for it in group_items]
        flagged = [r for r in reasons if r]
        excluded_reason = max(set(flagged), key=flagged.count) if len(flagged) * 2 > len(group_items) else None

        group_id = f"g{lab}"
        for i, it, v in zip(idxs, group_items, verdicts):
            enriched[i] = it.model_copy(update={"group_id": group_id, **v})

        prices = [it.price for it in group_items]
        trusted = [enriched[i] for i in idxs if enriched[i].deal_label != "suspicious"]
        best = min(trusted or [enriched[i] for i in idxs], key=lambda x: x.price)
        name = _group_name([it.title for it in group_items])
        mismatch = _mismatch(query, name)
        groups.append(ProductGroup(
            group_id=group_id,
            name=name,
            offers=len(group_items),
            min_price=min(prices),
            median_price=round(median(prices), 2),
            max_price=max(prices),
            best_offer_url=best.url,
            image=best.image or next((it.image for it in group_items if it.image), None),
            best_offer_price=best.price,
            is_accessory=excluded_reason is not None,
            excluded_reason=excluded_reason,
            is_refurbished=is_used(group_items[0].title),
            relevance=_relevance(mismatch),
            history_key=history_key([it.title for it in group_items]),
            suspicious_offers=sum(1 for i in idxs if enriched[i].deal_label == "suspicious"),
        ))

    _flag_price_outliers(groups)

    # Rank: real products first; exact match to the query, then variants, then other
    # models; new before refurbished; more offers (more evidence) first.
    order = {"exact": 0, "variant": 1, "other": 2}
    groups.sort(key=lambda g: (
        g.is_accessory, order[g.relevance], g.is_refurbished, -g.offers, g.median_price,
    ))

    # propagate group-level exclusion to each offer
    flags = {g.group_id: g.is_accessory for g in groups}
    enriched = [it.model_copy(update={"is_accessory": flags[it.group_id]}) for it in enriched]
    return enriched, groups


def _flag_price_outliers(groups: List[ProductGroup]) -> None:
    """
    Cross-group check. The 'anchor' is the typical price of the product the user
    is looking for: the offer-weighted median of the relevant, non-accessory,
    new-condition groups. Any group priced under 20% of the anchor (a ₹229 skin
    or a ₹4,309 display when the phone costs ₹64,000), or over 5x it (a ₹17.8 lakh
    'iPhone 15 with TikTok'), is flagged as an outlier.
    """
    candidates = [g for g in groups if not g.is_accessory and not g.is_refurbished and g.relevance != "other"]
    if not candidates:
        candidates = [g for g in groups if not g.is_accessory]
    if not candidates:
        return
    weighted = sorted(p for g in candidates for p in [g.median_price] * g.offers)
    anchor = weighted[len(weighted) // 2]
    for g in groups:
        if g.is_accessory:
            continue
        if g.median_price < anchor * PRICE_OUTLIER_RATIO or g.median_price > anchor * PRICE_OUTLIER_HIGH_RATIO:
            g.is_accessory = True
            g.excluded_reason = "price outlier"
