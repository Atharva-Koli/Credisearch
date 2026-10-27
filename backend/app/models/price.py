from pydantic import BaseModel
from typing import List, Optional


class PriceItem(BaseModel):
    source: str
    title: str
    price: float
    currency: str
    url: str

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
    best_offer_price: float
    is_accessory: bool
    is_refurbished: bool = False
    suspicious_offers: int


class SearchResponse(BaseModel):
    query: str
    country: str
    results: List[PriceItem]
    groups: List[ProductGroup] = []
    sources_ok: List[str] = []
    sources_failed: List[str] = []
