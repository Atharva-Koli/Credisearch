"""
Credisearch ML layer.

Two models run on every search:

1. Product matching (matcher.py)
   Listings from different sellers describe the same product in different
   words ("Apple iPhone 15 (128 GB) - Black" vs "iPhone 15 128GB Black").
   We embed every title as a TF-IDF vector over character n-grams, compare
   titles with cosine similarity, and cluster near-duplicates into one
   product group. Hard attribute guards (storage size, variant words like
   Pro/Max, model numbers) stop "iPhone 15" and "iPhone 15 Pro" merging.

2. Deal scoring (deal_scorer.py)
   Inside each product group we estimate the "fair" market price with the
   median and measure spread with the Median Absolute Deviation (MAD), a
   robust statistic that a single fake listing cannot drag around. Each
   offer gets a 0-100 deal score and a label: great deal, fair, overpriced,
   or suspicious (too cheap to be genuine).

Everything is pure Python, so the serverless bundle stays tiny and cold
starts stay fast.
"""

from app.services.ml.pipeline import analyze

__all__ = ["analyze"]
