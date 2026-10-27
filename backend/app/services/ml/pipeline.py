from collections import defaultdict
from statistics import median
from typing import List, Tuple

from app.models.price import PriceItem, ProductGroup
from app.services.ml.matcher import cluster_titles, display_name, is_accessory
from app.services.ml.deal_scorer import score_group


def _group_name(titles: List[str]) -> str:
    """Most representative title (median length), with colour words stripped."""
    by_len = sorted(titles, key=len)
    return display_name(by_len[len(by_len) // 2])


def analyze(items: List[PriceItem], query: str) -> Tuple[List[PriceItem], List[ProductGroup]]:
    """
    Enrich offers with product-group ids, accessory flags and deal verdicts.
    Returns (enriched_items, groups) — groups sorted with real products first,
    then by number of offers.
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
        accessory_votes = sum(is_accessory(it.title, query) for it in group_items)
        group_is_accessory = accessory_votes * 2 > len(group_items)
        group_id = f"g{lab}"

        for i, it, v in zip(idxs, group_items, verdicts):
            enriched[i] = it.model_copy(update={
                "group_id": group_id,
                "is_accessory": group_is_accessory,
                **v,
            })

        prices = [it.price for it in group_items]
        trusted = [enriched[i] for i in idxs if enriched[i].deal_label != "suspicious"]
        best = min(trusted or [enriched[i] for i in idxs], key=lambda x: x.price)
        groups.append(ProductGroup(
            group_id=group_id,
            name=_group_name([it.title for it in group_items]),
            offers=len(group_items),
            min_price=min(prices),
            median_price=round(median(prices), 2),
            max_price=max(prices),
            best_offer_url=best.url,
            best_offer_price=best.price,
            is_accessory=group_is_accessory,
            suspicious_offers=sum(1 for i in idxs if enriched[i].deal_label == "suspicious"),
        ))

    groups.sort(key=lambda g: (g.is_accessory, -g.offers, g.median_price))
    return enriched, groups
