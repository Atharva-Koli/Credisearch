from pydantic import BaseModel
from typing import Dict, List, Optional


class PriceItem(BaseModel):
    source: str
    title: str
    price: float
    currency: str
    url: str
    image: Optional[str] = None           # product thumbnail URL from the source

    # ---- ML enrichment (filled by app.services.ml) ----
    group_id: Optional[str] = None
    is_accessory: bool = False
    deal_label: Optional[str] = None      # great_deal | fair | overpriced | suspicious | no_comparison
    deal_score: Optional[int] = None      # 0-100, higher = better deal
    fair_price: Optional[float] = None    # median price of the product group
    vs_fair_pct: Optional[float] = None   # % above(+)/below(-) fair price
    robust_z: Optional[float] = None


class ProductGroup(BaseModel):
    group_id: str
    name: str
    offers: int
    min_price: float
    median_price: float
    max_price: float
    best_offer_url: str
    image: Optional[str] = None             # thumbnail of the best offer (or any offer)
    best_offer_price: float
    is_accessory: bool                       # True = not the product itself (hidden by hide_accessories)
    excluded_reason: Optional[str] = None    # accessory | spare part | add-on | price outlier
    is_refurbished: bool = False
    relevance: str = "exact"                 # exact | variant | other (different model)
    history_key: Optional[str] = None        # id for /history/{key}
    suspicious_offers: int


class SearchResponse(BaseModel):
    query: str
    country: str
    results: List[PriceItem]
    groups: List[ProductGroup] = []
    sources_ok: List[str] = []
    sources_failed: List[str] = []
    sources_disabled: Dict[str, str] = {}   # source -> reason, skipped by circuit breaker
    dropped_non_english: int = 0            # listings removed because the title isn't English
    history_enabled: bool = False           # True when the price logger (Redis) is configured
    cached: bool = False                    # served from cache: no API credits used
