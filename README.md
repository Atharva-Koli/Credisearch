# Credisearch
Your own search agent and price matcher.

**Live:** https://credisearchvercel.vercel.app (UI) · [/docs](https://credisearchvercel.vercel.app/docs) (API)

Search a product once. Credisearch pulls prices from multiple sources (SerpAPI + SearchAPI Google Shopping), **groups identical listings with ML**, and tells you which offers are real deals and which look too good to be true.

## ML layer (`backend/app/services/ml/`)

| Step | Technique | Why |
|---|---|---|
| **Match** | TF-IDF vectors over character 3–4-grams, cosine similarity, threshold clustering (union-find) | Sellers word the same product differently ("Apple iPhone 15 (128 GB) - Black" vs "iPhone 15 128GB Blue"). Char n-grams are robust to spacing, punctuation and typos. |
| **Guard** | Hard attribute rules: storage size, variant words (Pro/Max/Plus/Ultra), model numbers and codes (S24 vs S23, XM5 vs XM4) | Text similarity alone would merge "iPhone 15" with "iPhone 15 Pro". Domain rules keep precision high. |
| **Filter** | Accessory detection (case, cover, protector… unless you searched for one) | A ₹499 case shouldn't appear as the "cheapest iPhone". |
| **Price** | Median as fair price, Median Absolute Deviation (MAD) as spread, robust z-score | One fake listing can't drag the median the way it drags a mean. |
| **Judge** | 0–100 deal score + label: `great_deal`, `fair`, `overpriced`, `suspicious` | "Suspicious" = under 55% of the fair price, which is usually a scam, refurb or mislisting. The best offer shown always skips suspicious ones. |

Pure Python, with no numpy or torch, so the Vercel function stays tiny and cold starts stay fast.

### Roadmap
- **Price history + "buy now or wait"**: log prices per product group daily (Vercel Postgres/KV), then forecast drops.
- Swap TF-IDF for sentence embeddings once there's labelled match data to evaluate against.

## Run locally
```bash
cd backend
pip install -r requirements.txt
echo "SERPAPI_KEY=...\nSEARCHAPI_KEY=..." > .env
uvicorn app.main:app --reload     # UI at http://localhost:8000
python -m pytest tests -q         # ML tests
```

## API
`GET /search?q=iphone 15&limit=10&hide_accessories=true` returns `results` (offers with `group_id`, `deal_label`, `deal_score`, `vs_fair_pct`), `groups` (one per product with min/median/max and the best trusted offer), and `sources_ok` / `sources_failed`. If one source is down, the search still works with the other.

## Deployment
Vercel, root directory `backend`. Required environment variables: `SERPAPI_KEY`, `SEARCHAPI_KEY`.
