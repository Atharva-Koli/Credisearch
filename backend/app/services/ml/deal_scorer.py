"""
Deal scoring with robust statistics.

For a group of offers on the same product:
  fair price  = median
  spread      = 1.4826 * MAD   (MAD = median absolute deviation; the 1.4826
                factor makes it comparable to a standard deviation)
  robust z    = (price - median) / spread

Median and MAD ignore outliers, so one fake ₹999 "iPhone" cannot pull the
fair price down the way a mean would.
"""

from statistics import median
from typing import Dict, List

SUSPICIOUS_RATIO = 0.55   # cheaper than 55% of fair price => too good to be true
GREAT_DEAL_PCT = -0.07    # 7%+ below fair price
OVERPRICED_PCT = 0.10     # 10%+ above fair price
MIN_GROUP_FOR_STATS = 2


def _mad(values: List[float], center: float) -> float:
    return median(abs(v - center) for v in values)


def score_group(prices: List[float]) -> List[Dict]:
    """Return one verdict dict per price, same order as input."""
    if len(prices) < MIN_GROUP_FOR_STATS:
        return [{
            "deal_label": "no_comparison",
            "deal_score": None,
            "fair_price": None,
            "vs_fair_pct": None,
            "robust_z": None,
        } for _ in prices]

    fair = median(prices)
    spread = 1.4826 * _mad(prices, fair)
    # floor the spread so tiny groups with identical prices don't explode
    spread = max(spread, fair * 0.03)

    out = []
    for p in prices:
        pct = (p - fair) / fair
        z = (p - fair) / spread

        if p < fair * SUSPICIOUS_RATIO:
            label = "suspicious"
        elif pct <= GREAT_DEAL_PCT:
            label = "great_deal"
        elif pct >= OVERPRICED_PCT:
            label = "overpriced"
        else:
            label = "fair"

        # 0-100: 50 = fair price, higher = cheaper. Squashed z-score.
        score = 50 - 25 * max(-2.0, min(2.0, z))
        if label == "suspicious":
            score = min(score, 20)  # cheap but untrustworthy is not a good deal

        out.append({
            "deal_label": label,
            "deal_score": round(score),
            "fair_price": round(fair, 2),
            "vs_fair_pct": round(pct * 100, 1),
            "robust_z": round(z, 2),
        })
    return out
