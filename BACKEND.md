# Credisearch: Backend Logic

How the backend works, end to end: what happens on a search, what every module does, the data shapes, the endpoints, where data is stored, and how it's deployed.

- **ML details** (matching, deal scoring, outlier removal) → [ML.md](ML.md)
- **Running, deploying, troubleshooting** → [DEV.md](DEV.md)

---

## Contents
1. [Big picture](#1-big-picture)
2. [The life of one search](#2-the-life-of-one-search)
3. [Module by module](#3-module-by-module)
4. [Data models](#4-data-models)
5. [API endpoints](#5-api-endpoints)
6. [Price sources and normalisation](#6-price-sources-and-normalisation)
7. [Reliability: circuit breaker, cache, failure handling](#7-reliability)
8. [Price history storage](#8-price-history-storage)
9. [Daily cron job](#9-daily-cron-job)
10. [Configuration (env vars)](#10-configuration)
11. [Deployment on Vercel](#11-deployment-on-vercel)
12. [Tests](#12-tests)
13. [Known limitations and extension points](#13-known-limitations-and-extension-points)

---

## 1. Big picture

```
                       ┌──────────────────────────────────────────────┐
  Browser (UI at "/")  │            FastAPI app (Vercel function)      │
  ───────────────────► │                                              │
   GET /search?q=...   │  routes/search.py ──► api_clients/*  ───────────►  SerpAPI / Serper / SearchAPI
                       │        │                (one per source)      │      (Google Shopping)
                       │        ▼                                      │
                       │  aggregator.py  (raw JSON → PriceItem, INR)   │
                       │        ▼                                      │
                       │  English filter                               │
                       │        ▼                                      │
                       │  services/ml/  (match → score → clean → rank) │
                       │        ▼                                      │
                       │  price_log.py ───────────────────────────────────►  Upstash Redis (history)
                       │        ▼                                      │
                       │  cache.py (5 min) ──► JSON response           │
                       └──────────────────────────────────────────────┘
  Vercel Cron (daily) ──► GET /cron/log-prices ──► same pipeline for tracked products
```

- **Stack:** Python 3.12, FastAPI, Pydantic v2, httpx. No database driver or ML library is needed: the ML is pure Python and Redis is reached over HTTPS.
- **Size:** about 1,300 lines of Python in `backend/app/`, plus one HTML file for the UI.
- **Stateless by design:** every request can run on a fresh serverless instance. The only state is the in-memory cache and circuit breaker (both best-effort) and Redis (durable).

---

## 2. The life of one search

`GET /search?q=iphone 15 128gb&limit=40&hide_accessories=true`

| # | Step | Where | What happens |
|---|---|---|---|
| 1 | Validate | FastAPI / `search_prices()` | `q` must be ≥ 2 characters, `limit` 1–40. Bad input → **422**. |
| 2 | Cache check | `cache.get_cache()` | Key = `query(lowercased):country:limit:hide_accessories`. A hit returns immediately (no API credits used). |
| 3 | Pick sources | `active_sources()` | Every source with a key set **and** not switched off by the circuit breaker. None available → **503**. |
| 4 | Fetch | `run_search()` → `api_clients/*.search_products()` | Sources are called one after another, each in its own `try`. A source that errors, or returns `{"error": ...}` with HTTP 200, is recorded as failed (and may trip the circuit breaker); the others carry on. |
| 5 | Normalise | `aggregator.normalize_*_results()` | Up to 20 offers per source become `PriceItem`s: title, **price in INR**, URL, image, source. Foreign currencies are converted or dropped. |
| 6 | All failed? | `run_search()` | If no source succeeded → **502**. |
| 7 | English filter | `matcher.is_english()` | Titles that are more than 15% non-Latin letters (e.g. Hindi) are dropped; the count is returned as `dropped_non_english`. |
| 8 | ML | `ml.pipeline.analyze()` | Groups identical products, scores every offer, flags accessories/parts/outliers, sets relevance, ranks groups. See ML.md. |
| 9 | Log | `price_log.log_search()` | If Redis is configured: one event + one daily price point per relevant product. Never blocks or breaks the search if it fails. |
| 10 | Filter | `search_prices()` | With `hide_accessories=true`, excluded offers and groups are removed. |
| 11 | Trim & respond | `search_prices()` | Offers sorted cheapest first, trimmed to `limit`; **groups are not trimmed**. The ML runs on all offers *before* trimming, because the statistics need the full picture. |
| 12 | Cache store | `cache.set_cache()` | The response is kept for 5 minutes. |

Typical latency is 2–5 s, almost all of it waiting for the three external APIs (called sequentially). The ML itself takes milliseconds.

---

## 3. Module by module

```
backend/
├── app/
│   ├── main.py
│   ├── core/config.py
│   ├── models/price.py
│   ├── api/routes/search.py
│   ├── api/routes/history.py
│   ├── services/
│   │   ├── api_clients/serpapi.py · serper.py · searchapi.py · rapidapi.py (empty)
│   │   ├── aggregator.py
│   │   ├── source_health.py
│   │   ├── cache.py
│   │   ├── price_log.py
│   │   ├── ml/matcher.py · deal_scorer.py · pipeline.py
│   │   └── scraping/ (empty placeholders)
│   ├── utils/helpers.py (empty)
│   └── static/index.html · favicon-32.png · apple-touch-icon.png
├── tests/test_ml.py
├── vercel.json
├── requirements.txt
└── .env.example
```

| Module | Responsibility | Key functions |
|---|---|---|
| `main.py` | Creates the FastAPI app, mounts the routers, serves the UI and icons | `ui()` → `/`, `favicon()`, `apple_touch_icon()`, `health_check()` |
| `core/config.py` | Reads the three price-source keys from the environment (and `.env` locally via python-dotenv) | `SERPAPI_KEY`, `SERPER_API_KEY`, `SEARCHAPI_KEY` |
| `models/price.py` | Pydantic models for everything the API returns | `PriceItem`, `ProductGroup`, `SearchResponse` |
| `routes/search.py` | The `/search` endpoint and the shared search pipeline | `active_sources()`, `run_search()`, `search_prices()` |
| `routes/history.py` | Price-history endpoints and the daily cron | `list_history()`, `product_history()`, `recent_events()`, `cron_log_prices()` |
| `api_clients/*.py` | One thin HTTP client per source. Raise if the key is missing or the HTTP status is an error; return raw JSON | `search_products(query, country)` |
| `aggregator.py` | Turns each source's JSON into `PriceItem`s in INR; extracts thumbnails | `normalize_serpapi_results()`, `normalize_serper_results()`, `normalize_searchapi_results()`, `_price_to_inr()`, `_image()` |
| `source_health.py` | Circuit breaker: switches off sources that are out of credits or have a bad key | `is_available()`, `record_failure()`, `disabled_sources()` |
| `cache.py` | 5-minute in-memory response cache | `get_cache()`, `set_cache()` |
| `price_log.py` | Event log + daily price history in Upstash Redis (REST API) | `log_search()`, `get_history()`, `list_products()`, `recent_events()` |
| `ml/matcher.py` | Title cleaning, attribute extraction, TF-IDF, similarity, clustering, accessory/part/English checks, history keys | `normalize()`, `identity_tokens()`, `extract_attributes()`, `cluster_titles()`, `accessory_reason()`, `is_english()`, `canonical_key()`, `display_name()` |
| `ml/deal_scorer.py` | Median/MAD fair price and deal labels per group | `score_group()` |
| `ml/pipeline.py` | Runs matching + scoring, builds groups, flags price outliers, ranks | `analyze()`, `_mismatch()`, `_flag_price_outliers()`, `history_key()` |
| `static/index.html` | The whole UI: plain HTML/CSS/JS, no build step; calls `/search` and `/history/{key}` | `render()`, `card()`, `loadHistory()`, `sparkline()` |

`aggregator.merge_price_results()` is an older helper that's no longer used by the route (the route concatenates sources and lets the ML sort); it's kept for backwards compatibility.

---

## 4. Data models

### `PriceItem`: one offer from one seller
| Field | Type | Set by | Meaning |
|---|---|---|---|
| `source` | str | aggregator | `serpapi`, `searchapi`, or `serper · <seller>` (e.g. `serper · Flipkart`) |
| `title` | str | aggregator | Listing title as the seller wrote it |
| `price` | float | aggregator | Price in **INR** |
| `currency` | str | aggregator | Always `"INR"` after normalisation |
| `url` | str | aggregator | Link to the offer |
| `image` | str \| null | aggregator | Thumbnail URL |
| `group_id` | str | ML | Which product group the offer belongs to (`g0`, `g1`, …) |
| `is_accessory` | bool | ML | True if its group was excluded (accessory, part, add-on, price outlier) |
| `deal_label` | str | ML | `great_deal` · `fair` · `overpriced` · `suspicious` · `no_comparison` |
| `deal_score` | int \| null | ML | 0–100, higher = better deal (null when the group has 1 offer) |
| `fair_price` | float \| null | ML | Median price of its group |
| `vs_fair_pct` | float \| null | ML | % above (+) or below (−) the fair price |
| `robust_z` | float \| null | ML | Robust z-score (see ML.md §4.3) |

### `ProductGroup`: one real product (many offers)
| Field | Type | Meaning |
|---|---|---|
| `group_id` | str | Id used by `PriceItem.group_id` |
| `name` | str | Representative title with colours removed |
| `offers` | int | Number of offers in the group |
| `min_price` / `median_price` / `max_price` | float | Price range; median = fair price |
| `best_offer_url`, `best_offer_price` | str, float | Cheapest offer that is **not** suspicious |
| `image` | str \| null | Thumbnail of the best offer (or any offer) |
| `is_accessory` | bool | Excluded from the main results |
| `excluded_reason` | str \| null | `accessory` · `spare part` · `add-on` · `price outlier` |
| `is_refurbished` | bool | Refurbished / renewed / used listing |
| `relevance` | str | `exact` · `variant` · `other` compared with the query |
| `history_key` | str | Spelling-independent product id for `/history/{key}` (e.g. `apple-iphone-15-128gb`) |
| `suspicious_offers` | int | How many offers were labelled suspicious |

### `SearchResponse`
| Field | Meaning |
|---|---|
| `query`, `country` | Echo of the request |
| `results` | Offers (`PriceItem`), cheapest first, trimmed to `limit` |
| `groups` | All product groups (`ProductGroup`), best first |
| `sources_ok` / `sources_failed` | Which sources returned data / errored on this request |
| `sources_disabled` | Sources currently skipped by the circuit breaker, with the reason |
| `dropped_non_english` | Listings removed by the English filter |
| `history_enabled` | True when Redis is configured (the UI then loads history) |

---

## 5. API endpoints

Interactive docs (Swagger) are at `/docs`.

| Method & path | Purpose | Responses |
|---|---|---|
| `GET /` | The search UI (HTML) | 200 |
| `GET /search` | Search prices + ML | 200 · 422 bad input · 502 all sources failed · 503 no source available |
| `GET /health` | Liveness check → `{"status": "ok"}` | 200 |
| `GET /history` | Every product with stored history → `{enabled, products: {key: name}}` | 200 |
| `GET /history/{key}` | Daily points for one product + summary (`days_tracked`, `first_date`, `lowest_min`, `lowest_date`, `latest_fair`) | 200 · 503 if Redis isn't configured |
| `GET /history/events/recent?limit=50` | The latest search events (event log) | 200 |
| `GET /cron/log-prices` | Daily job (hidden from docs); needs `Authorization: Bearer <CRON_SECRET>` when the secret is set | 200 · 401 |
| `GET /tracker` | Tracked and previously tracked products, each with its main product's daily timeline, plus `can_edit` | 200 |
| `POST /tracker` `{"q": "pixel 10"}` | Start tracking a product and record today's price immediately; needs `X-Admin-Token` | 200 · 401 · 403 (no `ADMIN_TOKEN` set) · 409 (duplicate or limit of 10) |
| `DELETE /tracker?q=...` | Stop tracking; **the history is kept**; needs `X-Admin-Token` | 200 · 401 · 404 |
| `GET /favicon.ico`, `/favicon-32.png`, `/apple-touch-icon.png` | Logo icons | 200 |

### `/search` parameters
| Param | Default | Rules | Effect |
|---|---|---|---|
| `q` | required | ≥ 2 characters | Product to search |
| `country` | `IN` | Google country code | Sent as `gl` to every source |
| `limit` | `10` | 1–40 | Number of offers in `results` (the UI asks for 40) |
| `hide_accessories` | `false` | bool | Drop excluded offers and groups |

### Example response (trimmed)
```json
{
  "query": "iphone 15 128gb",
  "country": "IN",
  "results": [
    {"source": "serper · Flipkart", "title": "Apple iPhone 15 (Blue, 128 GB)", "price": 65999.0,
     "currency": "INR", "url": "https://…", "image": "https://…", "group_id": "g3",
     "is_accessory": false, "deal_label": "fair", "deal_score": 67, "fair_price": 68699.5,
     "vs_fair_pct": -3.9, "robust_z": -0.66}
  ],
  "groups": [
    {"group_id": "g3", "name": "Apple iPhone 15 (128 GB)", "offers": 6, "min_price": 21999.0,
     "median_price": 68699.5, "max_price": 79999.0, "best_offer_url": "https://…",
     "image": "https://…", "best_offer_price": 65999.0, "is_accessory": false,
     "excluded_reason": null, "is_refurbished": false, "relevance": "exact",
     "history_key": "apple-iphone-15-128gb", "suspicious_offers": 1}
  ],
  "sources_ok": ["serpapi", "serper", "searchapi"],
  "sources_failed": [],
  "sources_disabled": {},
  "dropped_non_english": 0,
  "history_enabled": false
}
```

---

## 6. Price sources and normalisation

All three are Google Shopping APIs. Each search costs **1 credit per active source**.

| Source | Client | Request | Response list | Free allowance |
|---|---|---|---|---|
| **SerpAPI** | `serpapi.py` | `GET serpapi.com/search.json`, `engine=google_shopping`, `gl`, `hl=en` | `shopping_results[]` | 250 searches/month, refills monthly |
| **Serper.dev** | `serper.py` | `POST google.serper.dev/shopping`, header `X-API-KEY`, body `{q, gl, hl, num: 20}` | `shopping[]` | 2,500 on signup, then ~$1 per 1,000 |
| **SearchAPI** | `searchapi.py` | `GET searchapi.io/api/v1/search`, `engine=google_shopping`, `gl` (lower-case), `hl=en` | `shopping_results[]` | One-off free credits, no refill |

All clients use a **20-second timeout** and raise on a missing key or an HTTP error status.

### Normalisation rules (`aggregator.py`)
Every offer must end up with a title, a link and a positive **INR** price, or it's skipped. Titles first go through `fix_mojibake()`, which repairs text that was UTF-8 but arrived decoded as Latin-1 (e.g. Hindi showing up as "à¤…"). Otherwise those broken characters look like Latin letters and slip past the English-only filter.

| Source | Price fields used | Currency handling |
|---|---|---|
| SerpAPI | `price` string first, else `extracted_price` | `₹` → as is; `$` → × `USD_TO_INR` (83); with only `extracted_price`, values under 2,000 are assumed USD |
| SearchAPI | `extracted_price`, else digits from `price` | `$` in the price string → × 83; `€ £ ¥` → dropped |
| Serper | `price` string via `_price_to_inr()` | `₹`/`Rs` → as is; `$` → × 83; `€ £ ¥` → dropped; no symbol → assumed INR (we request `gl=in`) |

- **Link field:** SerpAPI `product_link`; SearchAPI `link` or `product_link`; Serper `link`.
- **Image:** `thumbnail` (SerpAPI/SearchAPI) or `imageUrl` (Serper); only `https://` or `data:image/` URLs are kept.
- **Seller name:** Serper returns `source` (the store), stored as `serper · <store>`.

**Why `gl` matters:** SearchAPI ignores a `country` parameter and needs `gl`. Before that fix it silently returned US listings in dollars, which showed up as "₹309 iPhones".

---

## 7. Reliability

### Independent sources
Each source is fetched in its own `try`. One failure is logged, recorded in `sources_failed`, and the search continues with the rest. Only when **every** source fails does the request return 502.

### Circuit breaker (`source_health.py`)
| Trigger | Action |
|---|---|
| HTTP **401 / 402 / 403 / 429** | Source switched off for **6 hours** (`COOLDOWN_SECONDS`) |
| Error text matching credit / quota / run out / limit reached / exceeded / plan / subscription / invalid api key / unauthorized | Switched off for 6 hours |
| HTTP 200 with `{"error": "..."}` in the body | Treated as a failure, then checked against the words above |
| Timeouts, 5xx, a missing key ("not configured") | **Not** switched off: these are usually temporary or a config issue |

The state lives in process memory, so a new Vercel instance starts fresh. The worst case is one wasted failed call per cold start, and failed calls don't use credits. To retire a source permanently, delete its key in Vercel.

### Cache (`cache.py`)
- In-memory dict: key → (timestamp, response), **TTL 5 minutes**.
- Saves credits when the same search is repeated (e.g. toggling the UI or reloading).
- Per serverless instance and lost on cold start, so it's an optimisation, not a guarantee.

### Logging never breaks a search
`price_log._pipeline()` catches every exception, logs a warning and returns `None`. A Redis outage only means that search isn't recorded.

---

## 8. Price history storage

**Upstash Redis**, used through its REST API with httpx: one `POST {URL}/pipeline` request sends several commands at once. If the Redis env vars are missing, every logging function quietly does nothing and `history_enabled` is `false`.

| Key | Redis type | Content | Limit |
|---|---|---|---|
| `events` | List (newest first) | One JSON event per search: `{ts, origin: "user"\|"cron", q, sources, products: [{key, fair, min, offers}]}` | Trimmed to 5,000 |
| `products` | Hash | `history_key → product name` | — |
| `tracked` | Hash | query → `{added_at}`: the list the daily job searches (editable from the UI) | Max 10 |
| `tracked_removed` | Hash | query → `{added_at, removed_at}`: stopped products; their history stays | — |
| `track_main` | Hash | query → history key of its main (exact-match) product, set by tracked searches | — |
| `tracked_init` | String | Set once the list has been seeded from `TRACKED_PRODUCTS` / defaults | — |
| `price:<history_key>` | Hash | `YYYY-MM-DD → {fair, min, max, offers, best_url}`; `min` is the best **trusted** price (suspicious offers excluded) | One field per day; the latest search that day wins |

**What gets logged per search:** up to 5 groups that are not excluded, not refurbished and not "other" relevance. If two groups map to the same product key, the one with more offers wins.

**`history_key`:** built by `product_key()`: only the identifying words at the start of the title (first 5 meaningful words: brand, model, variant), colours removed, "Apple" dropped before iPhone/iPad/MacBook, glued names split ("reno15" → "reno 15"), plus storage. Seller junk at the end of a title never reaches the key. The most common key among a group's titles is used, so "OPPO Reno 15 Pro Mini 5G Glacier 256GB without offers" and "Oppo Reno 15 Pro Mini 5G Cocoa Brown 256GB" share `oppo-reno-15-pro-mini-256gb`.

**Reading:** `get_history(key)` returns the points sorted by date; `/history/{key}` adds a summary. The UI draws a sparkline of the fair price when there are 2+ days.

**Planned:** an importer that writes Apify's multi-year Amazon.in history into the same `price:<key>` hashes (see ML.md §10).

---

## 8b. Price tracker (editable tracked list)

The **Price tracker** tab shows what's being tracked and a timeline for each product.

- **The list lives in Redis** (`tracked`), so it's changed from the UI or the API, with no redeploy. The first time it's read, it's seeded from `TRACKED_PRODUCTS` or the defaults in `history.py`.
- **Adding** (`POST /tracker`) checks the 10-product limit and duplicates, then immediately runs one search with Serper (`origin="track"`) so today's price appears at once.
- **Stopping** (`DELETE /tracker`) moves the product to `tracked_removed`. **No price data is ever deleted**; the timeline keeps showing it under "No longer tracked". Tracking it again brings it back with the same history.
- **Which product is "the" product for a query?** Tracked searches store the best-ranked exact match in `track_main` (e.g. "oneplus 15r" → `oneplus-15-r-256gb`, not the OnePlus 15). If that's missing, the closest key starting with `product_key(query)` is used.
- **Editing is admin-only**: `ADMIN_TOKEN` env var, sent as the `X-Admin-Token` header and compared in constant time. Each tracked product costs a Serper credit every day, and the site is public. Without `ADMIN_TOKEN` the tracker is read-only. The UI remembers the key in the browser's local storage.
- **Timeline** (`GET /tracker` → `price_log.get_timeline()`): a few pipelined Redis calls return every tracked and stopped product with its daily points (fair price and best trusted price) and a summary (latest, lowest and when, days of data).

## 9. Daily cron job

- **Schedule:** `backend/vercel.json` → `/cron/log-prices` at **02:30 UTC (08:00 IST)** every day.
- **Auth:** when `CRON_SECRET` is set, Vercel sends `Authorization: Bearer <CRON_SECRET>`; any other caller gets 401.
- **Products:** the editable list in Redis (see 8b). It's seeded the first time from `TRACKED_PRODUCTS` (comma-separated) or the defaults in `history.py`: iPhone 17, iPhone 16, OnePlus 15R, Oppo Reno 15 Pro Mini 5G, Samsung Galaxy S25.
- **Sources:** **Serper only** (falls back to SerpAPI if Serper is unavailable), so it doesn't use up SerpAPI's 250/month. About 5 credits/day ≈ 150/month.
- **Runs the same `run_search()` pipeline** with `origin="cron"`, so cron data and user data are identical in shape.
- **Time guard:** stops starting new products after 50 s to stay inside the function time limit.
- **Returns** `{ok, source, logged: [...], failed: [...]}`. Without Redis it returns `{ok: false, reason: ...}` and does nothing.

Manual test: `curl -H "Authorization: Bearer $CRON_SECRET" https://credisearchvercel.vercel.app/cron/log-prices`

---

## 10. Configuration

All optional, but search needs at least one source key. Template: `backend/.env.example`.

| Variable | Used by | Purpose |
|---|---|---|
| `SERPAPI_KEY` | `serpapi.py` | SerpAPI source |
| `SERPER_API_KEY` | `serper.py` | Serper source (also the cron's source) |
| `SEARCHAPI_KEY` | `searchapi.py` | SearchAPI source |
| `KV_REST_API_URL` + `KV_REST_API_TOKEN` (or `UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN`) | `price_log.py` | Price history; added automatically by Vercel's Upstash integration |
| `CRON_SECRET` | `history.py` | Protects the cron endpoint |
| `ADMIN_TOKEN` | `history.py` | Password for adding/stopping tracked products from the UI |
| `TRACKED_PRODUCTS` | `history.py` | Seeds the tracked list the first time (after that, edit it in the UI) |
| `APIFY_TOKEN` *(planned)* | importer | Backfilling Amazon.in history |

Locally they come from `backend/.env` (loaded by python-dotenv). On Vercel they're set in Project → Settings → Environment Variables, and **a redeploy is required** after changing them.

---

## 11. Deployment on Vercel

- **Project:** `credisearch_vercel`, connected to `Atharva-Koli/Credisearch`; **every push to `main` deploys automatically.**
- **Root directory:** `backend`. Vercel detects FastAPI from `app/main.py` and runs it as a Python serverless function; `requirements.txt` is installed at build time.
- **Static files:** `app/static/*` are read from disk by the routes in `main.py` (bundled with the function).
- **`vercel.json`:** only defines the cron; everything else is zero-config.
- **Dependencies:** `fastapi`, `uvicorn[standard]`, `httpx`, `python-dotenv`, `pydantic`, `beautifulsoup4` (the last one is unused for now). No numpy/torch, which keeps the bundle small and cold starts fast.

---

## 12. Tests

`backend/tests/test_ml.py`: 31 tests, run with `cd backend && python -m pytest tests -q`. They need no API keys and use no credits.

| Area | What's covered |
|---|---|
| Matching | Wording/colour variants merge; Pro/Plus/Mini, storage, generations (15 vs 14, Reno15 vs Reno 14), model codes (S24/S23, XM5/XM4), `+` = Plus, `8/128` and `12+512GB`, Reno15c, spec-heavy titles, refurbished, product vs its part |
| Cleaning | English detection, accessory / spare-part / add-on rules, price outliers, off-topic results, relevance tiers, display names, history keys |
| Scoring | Deal labels on the worked example, best trusted offer skips suspicious |
| Reliability | Circuit breaker on HTTP 429 and on "run out of credits" messages; not on timeouts/500s |
| History | Logger writes events and daily points through an in-memory fake Redis; trusted lowest price; key dedupe |
| Tracker | Seeding, add/remove, duplicates, the 10-item limit, history kept after stopping and restored on re-tracking, main-product selection, admin-key protection on the API |

---

## 13. Known limitations and extension points

**Limitations**
- Sources are fetched **sequentially**; running them in parallel (threads or `httpx.AsyncClient`) would cut latency to roughly the slowest single source.
- Cache and circuit breaker are per instance and in memory.
- Fixed USD→INR rate (83).
- The word lists (variants, accessories, parts, colours) are English and electronics-focused.
- `rapidapi.py`, `scraping/`, `utils/helpers.py` and `core/logging.py` are empty placeholders from the original project.

**How to extend**
- **New price source:** add `api_clients/<name>.py` (`search_products(query, country) -> dict`), a `normalize_<name>_results()` in `aggregator.py` returning INR `PriceItem`s, the key in `config.py` and `.env.example`, and register it in `active_sources()`.
- **New matching rule:** add the word/regex in `matcher.py`, add a failing example to `tests/test_ml.py`, make it pass.
- **New history consumer (e.g. "buy now or wait"):** read `price:<key>` via `price_log.get_history()`; no schema change needed.
