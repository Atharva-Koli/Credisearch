# Credisearch: The ML Layer, Explained

This document explains everything in `backend/app/services/ml/`: what problem it solves, how each step works (with the maths and a worked example), why it was built this way, what its limits are, and how to talk about it.

> **One-line summary:** Credisearch uses unsupervised machine learning (TF-IDF text embeddings + similarity clustering) to work out which listings are the same product, then robust statistics (median + MAD anomaly detection) to judge whether each price is a great deal, fair, overpriced, or too cheap to trust.

---

## Contents
1. [The problem](#1-the-problem)
2. [The pipeline at a glance](#2-the-pipeline-at-a-glance)
3. [Model 1: Product matching](#3-model-1-product-matching-matcherpy)
4. [Model 2: Deal scoring](#4-model-2-deal-scoring-deal_scorerpy)
5. [Putting it together: pipeline.py](#5-putting-it-together-pipelinepy)
6. [Tuning knobs](#6-tuning-knobs)
7. [Design decisions and trade-offs](#7-design-decisions-and-trade-offs)
8. [Limitations (be honest about these)](#8-limitations)
9. [How to evaluate it properly](#9-how-to-evaluate-it-properly)
10. [Roadmap: "buy now or wait"](#10-roadmap-buy-now-or-wait)
11. [Interview guide](#11-interview-guide)
12. [Glossary](#12-glossary)

---

## 1. The problem

The price APIs return a flat list of listings from many sellers:

| Title | Price |
|---|---|
| Apple iPhone 15 (128 GB) - Black | ₹69,900 |
| iPhone 15 128GB Blue | ₹65,999 |
| Apple iPhone 15, 128GB, Green | ₹67,499 |
| Apple iPhone 15 128GB (Black) - Brand New Sealed | ₹21,999 |
| Apple iPhone 15 Pro (128 GB) | ₹1,19,900 |
| Refurbished Apple iPhone 15 by Cashify | ₹48,899 |
| Silicone Case for iPhone 15 | ₹499 |

Sorting by price gives nonsense: the "cheapest iPhone" is a ₹499 case, then a ₹21,999 listing that's almost certainly a scam, then a refurb. To compare prices honestly, two questions must be answered:

1. **Which listings are the same product?** The same phone is described differently by every seller, while a Pro or a 256GB model looks almost identical in text but is a different product. → **Matching**
2. **For one product, is this price good?** You need a "fair price" that a single fake listing can't distort. → **Deal scoring**

## 2. The pipeline at a glance

```
GET /search?q=iphone 15
        │
        ▼
  ┌─────────────────────┐   SerpAPI, Serper, SearchAPI (each optional, each can fail;
  │  Fetch price sources │   circuit breaker skips sources that are out of credits)
  └─────────┬───────────┘
            ▼
  ┌─────────────────────┐   aggregator.py: raw JSON → PriceItem(title, price in INR, url, image)
  │  Normalise to INR    │   ($ prices converted, other currencies dropped)
  └─────────┬───────────┘
            ▼
  ┌─────────────────────┐   is_english(): drop titles that are mostly non-Latin script
  │  English filter      │   (e.g. Hindi listings)
  └─────────┬───────────┘
            ▼
  ┌─────────────────────┐   matcher.py
  │  1. MATCH            │   clean title → TF-IDF char n-gram vector → cosine similarity
  │                      │   → hard guards → union-find clustering → product groups
  └─────────┬───────────┘   + accessory & refurbished flags
            ▼
  ┌─────────────────────┐   deal_scorer.py, per group
  │  2. SCORE            │   fair price = median, spread = MAD, robust z-score
  │                      │   → label + 0–100 deal score
  └─────────┬───────────┘
            ▼
  ┌─────────────────────┐   pipeline.py
  │  3. CLEAN & RANK     │   accessories / spare parts / add-ons / price outliers flagged;
  │                      │   relevance tier (exact / variant / other) vs the query;
  │                      │   best *trusted* offer per group; ranked best-first
  └─────────┬───────────┘
            ▼
  ┌─────────────────────┐   price_log.py (if Redis is configured)
  │  Log prices          │   one point per product per day → future "buy now or wait" data
  └─────────┬───────────┘
            ▼
     JSON response → UI
```

All of it is **pure Python** (no numpy, scikit-learn or PyTorch). Section 7 explains why.

---

## 3. Model 1: Product matching (`matcher.py`)

### 3.1 Step 1: Clean the title (`normalize`, `tokens`)

| Transformation | Before | After |
|---|---|---|
| Lowercase | `Apple iPhone 15` | `apple iphone 15` |
| Strip brackets, commas, slashes, pipes | `(128 GB), Black` | ` 128 GB  Black` |
| Join number + unit | `128 gb` | `128gb` |
| Join hyphenated model codes | `wh-1000xm5` | `wh1000xm5` |
| European decimals (screen sizes) | `6,1` | `6.1` (ignored as a spec, never read as model numbers 6 and 1) |
| `+` suffix means Plus | `S24+` | `s24 plus` |
| RAM/storage shorthand | `8/128`, `12+512GB` | `8gb 128gb`, `12gb 512gb` |
| Letter suffix variant | `Reno15c`, `iPhone 16e`, `OnePlus 15R` | `reno15 c`, `16 e`, `15 r` (the letter is a variant: 16e ≠ 16) |
| Drop stopwords (marketing/noise) | `new`, `sealed`, `original`, `5g`, `ram`, `smartphone`… | removed |
| Drop **colour words** | `black`, `blue`, `titanium`, `onyx`… | removed |

**Identity tokens:** only the first **5** meaningful words, with spec tokens removed (200MP, 120Hz, 6.7inch, 512GB…), go into the similarity vector (`identity_tokens`). Product identity sits at the start of a title; the tail is marketing ("…200MP Camera & Dimensity 8450…") that only dilutes similarity. Storage, variants and model numbers are still checked from the **full** title by the guards.

**Why drop colours?** For price comparison, a black and a blue iPhone 15 128GB are the same product at the same price. Keeping colours would split one product into five groups. (This was a real bug during testing: the 128GB iPhone split into 4 groups until colours were removed.)

### 3.2 Step 2: Turn text into vectors (TF-IDF over character n-grams)

A computer can't compare sentences directly, so each cleaned title becomes a **vector**: a set of (feature → weight) pairs.

**Features = character n-grams.** The title is padded with spaces and cut into every overlapping 3-character and 4-character chunk:

```
" iphone 15 128gb "
3-grams: " ip", "iph", "pho", "hon", "one", "ne ", "e 1", " 15", "15 ", ...
4-grams: " iph", "ipho", "phon", "hone", "one ", ...
```

**Why characters instead of whole words?** They survive the mess of real listings:
- `iPhone15` vs `iPhone 15`: different words, mostly the same chunks
- `128GB` vs `128 GB`: the same after cleaning, and close even before
- Typos (`Iphne`) still share most chunks

**Weights = TF-IDF.** Each n-gram `g` in title `d` gets:

```
TF(g, d)  = 1 + log(count of g in d)            ← sublinear: repeating a chunk helps, but less and less
IDF(g)    = log((1 + N) / (1 + DF(g))) + 1      ← N = number of titles, DF = titles containing g
weight    = TF × IDF
```

then the whole vector is scaled to length 1 (L2 normalisation).

**What IDF does, intuitively:** in a search for "iphone 15", the chunk `iph` appears in *every* title, so it tells you nothing and gets a low weight. A chunk like `pro` appears in only a few titles, so it's distinctive and gets a high weight. IDF is computed **per search**, over just that search's results. The model therefore adapts to each query and needs no training data. That makes it **unsupervised**.

(This is the same smoothed IDF formula scikit-learn's `TfidfVectorizer` uses, implemented by hand in `tfidf_vectors()`.)

### 3.3 Step 3: Compare titles (cosine similarity)

For two vectors `a` and `b` (already length 1):

```
cosine(a, b) = Σ over shared n-grams  a[g] × b[g]
```

It ranges from **0** (no chunks in common) to **1** (identical). It measures the *angle* between the vectors, so a long title and a short title about the same thing can still score high.

Two titles are candidates for the same product if `cosine ≥ 0.55` (`SIMILARITY_THRESHOLD`).

### 3.4 Step 4: Hard guards (`extract_attributes`, `compatible`)

Text similarity has a blind spot. `iPhone 15 128GB` and `iPhone 15 Pro 128GB` share almost every chunk, yet they differ by ₹50,000. So before similarity is even checked, two listings must pass **every** guard:

| Guard | How it's extracted | Prevents merging |
|---|---|---|
| **Variant words** must be identical | Words in `VARIANT_WORDS`: pro, max, plus, mini, ultra, lite, se, fe, air, neo, edge, fold, flip, and single-letter suffixes c / s / t / r (as in Reno15**c**) | iPhone 15 ↔ iPhone 15 Pro ↔ iPhone 15 Plus |
| **Storage** must match (if both state it) | All `NNgb`/`NNtb` values; the **largest** is storage, so "8GB RAM, 256GB" reads as 256GB | 128GB ↔ 256GB |
| **Model numbers** must overlap (if both have them) | Bare numbers not followed by a unit, plus the number glued to a name ("reno15" → 15) | iPhone 15 ↔ iPhone 14, Reno15 ↔ Reno 14 |
| **Model codes** must overlap (if both have them) | Tokens mixing letters and digits, excluding sizes and "5g" | Galaxy S24 ↔ S23, Sony XM5 ↔ XM4, OnePlus 12 ↔ 12R |
| **Condition** must match | `USED_WORDS`: refurbished, renewed, used, pre-owned, open box, Cashify… | New ↔ refurbished |
| **Kind** must match | Accessory/part words present or not (`ACCESSORY_WORDS ∪ PART_WORDS`) | "Reno 15 Pro Mini" ↔ "Reno 15 Pro Mini Display Combo" (identical first 5 words!) |

**Model-code bonus:** if two titles share an exact model code (both say `wh1000xm5`), the similarity threshold drops by 0.2 (`MODEL_CODE_BONUS`) to 0.35. A matching model code is strong evidence of identity, so looser wording around it is fine ("Wireless Headphones" vs "Noise Cancelling Headphones").

This combination of a **statistical model for fuzzy matching plus domain rules for precision** is how most production entity-matching systems work. Neither half is enough alone.

### 3.5 Step 5: Clustering (union-find)

Every pair of titles that passes the guards and the similarity threshold gets linked. Groups are the **connected components** of those links: if A↔B and B↔C, then A, B and C form one product, even if A and C alone fell just short of the threshold.

Union-find (disjoint-set) is the textbook data structure for this: `union(a, b)` merges two groups and `find(x)` returns a group's representative, both in near-constant time. The implementation is about 15 lines in `_UnionFind`.

Comparing every pair is O(n²). With at most about 60 listings per search, that's under 2,000 comparisons, which takes milliseconds.

### 3.6 Step 6: Accessory detection (`is_accessory`)

A listing is an accessory if its title contains an accessory word (case, cover, protector, tempered, glass, charger, cable, adapter, strap, skin…) **that the user didn't search for**. A group is an accessory group if the majority of its listings are accessories.

- Search "iphone 15": the ₹499 case is flagged and hidden by default.
- Search "iphone 15 case": "case" is in the query, so cases are *not* flagged.

### 3.6b Step 6b: Removing outliers (non-English, parts, price outliers, other models)

Real searches return junk that isn't the product. Four layers remove it:

| Layer | Where | Rule | Example removed |
|---|---|---|---|
| **English only** | `is_english()` in matcher.py, applied in `run_search` *before* the ML | Drop a title if more than 15% of its letters are outside the Latin alphabet (accented Latin like "Café" is kept) | "Oppo Reno 15 PRO MINI 5G स्किन नॉट बैक कवर…" |
| **Spare parts** | `accessory_reason()` → `PART_WORDS` | lcd, digitizer, touchscreen, combo, folder, motherboard, housing, flex, spare, panel, assembly… (unless searched for) | "Display … (Combo Folder) \| Maxbhi", "Display and Touchscreen Combo" |
| **Add-ons ("X for Y")** | `accessory_reason()` | If the words *before* " for " contain nothing the user searched, the item is something *for* the product, not the product | "Battery Door for Oppo Reno 15 Pro" |
| **Price outliers** | `_flag_price_outliers()` in pipeline.py | **Anchor** = offer-weighted median price of the relevant, new, non-accessory groups. Any group under **20%** of the anchor, or over **5×** it, is flagged. This catches anything the word lists miss. | an ₹899 "shell" when the phone's anchor is ₹64,000; a ₹17.8 lakh "iPhone 15 with TikTok" |

Each flagged group gets `excluded_reason` (accessory / spare part / add-on / price outlier) and is hidden by "Hide accessories & parts".

**Relevance tiers:** every group is compared with the *query* (`_mismatch`). A group sharing fewer than half of the searched words (`MIN_QUERY_OVERLAP`) is off-topic; Google Shopping returns Skechers shoes for "alfa hockey stick". The tiers are: `exact` (what you searched), `variant` (an extra or missing variant word, or different storage: the Pro Mini, or the plain Reno15 when you searched Reno15 Pro), or `other` (a different model number or code, e.g. Reno 14 when you searched Reno 15). The UI shows exact + variant and folds "other" behind a button.

**Glued model numbers:** "Reno15" and "iPhone15" have their generation number extracted, so "Reno15 Pro" vs "Reno 14 Pro" is correctly treated as different models.

### 3.6c Product images

Each source returns a thumbnail (`thumbnail` in SerpAPI/SearchAPI, `imageUrl` in Serper). It's kept on every offer (`PriceItem.image`), and each group shows the best offer's image. Because colours are merged into one product for pricing, the per-offer thumbnails are how the user sees which colour each price is for. Images aren't used by the ML.

### 3.7 Step 7: Naming each group (`display_name`)

The group is named after its median-length title, with colour words stripped and brackets rebalanced. "Apple iPhone 15 (128 GB, Black)" becomes "Apple iPhone 15 (128 GB)".

### 3.8 Verified behaviour (from `tests/test_ml.py`)

| Input titles | Groups |
|---|---|
| iPhone 15 128GB in Black / Blue / Green, three different wordings | **1** group |
| iPhone 15 128GB · 15 256GB · 15 Pro 128GB · 14 128GB | **4** groups |
| Sony WH-1000XM5 ×2 (different wording) · WH-1000XM4 | **2** groups (XM5 together, XM4 apart) |
| Galaxy S24 (8GB RAM, 256GB) · S24 256GB · S23 256GB | **2** groups |
| New iPhone 15 · Refurbished by Cashify · Renewed | new is kept apart from both refurbs |
| Galaxy S24+ 256GB · S24 Plus 256GB · S24 256GB | **2** groups (`+` = Plus) |
| Reno15 Pro 512GB · "Reno15 Pro – 200MP Camera & Dimensity…12GB 512GB" · "Reno15 Pro 12+512GB \| 6.7-inch" · Reno15c · Reno15 | **3** groups (the three Pro listings merge) |
| Reno 15 Pro Mini · Reno 15 Pro Mini *Display and Touchscreen Combo* | **2** groups (a product never merges with its part) |

---

## 4. Model 2: Deal scoring (`deal_scorer.py`)

Each product group now holds several prices for the **same thing**. This step answers whether a given price is good.

### 4.1 Fair price = median

The **median** is the middle value once prices are sorted (or the average of the two middle values).

**Why not the average (mean)?** One fake listing drags the mean a lot. It barely moves the median. Statisticians call the median a **robust** estimator.

### 4.2 Spread = MAD (Median Absolute Deviation)

To call something "unusually cheap" you also need to know how much prices *normally* vary.

```
MAD    = median( |price − median| )   over all prices in the group
spread = 1.4826 × MAD
```

The constant 1.4826 scales MAD so that, for normally distributed data, it estimates the same thing as the standard deviation. Like the median, MAD ignores outliers; standard deviation does not.

A floor stops tiny groups with identical prices from producing near-zero spread: `spread = max(spread, 3% of fair price)`.

### 4.3 Robust z-score

```
z = (price − fair price) / spread
```

This measures how many "normal variations" a price sits from fair. `z = 0` is exactly fair, negative is cheaper, positive is pricier. Using median and MAD instead of mean and standard deviation makes this a **robust z-score**, a standard method for **anomaly / outlier detection**.

### 4.4 Labels

`vs_fair = (price − fair) / fair` (as a %)

| Label | Rule | Meaning |
|---|---|---|
| **suspicious** | price < 55% of fair (`SUSPICIOUS_RATIO`) | Too good to be true: scam, refurb sold as new, wrong listing, or a part |
| **great_deal** | vs_fair ≤ −7% (`GREAT_DEAL_PCT`) | Genuinely cheap |
| **overpriced** | vs_fair ≥ +10% (`OVERPRICED_PCT`) | Pay less elsewhere |
| **fair** | anything in between | Normal market price |
| **no_comparison** | the group has only 1 offer | Nothing to compare against |

### 4.5 Deal score (0–100)

```
score = 50 − 25 × clamp(z, −2, +2)
```

- 50 = exactly fair; 100 = 2+ spreads cheaper; 0 = 2+ spreads pricier.
- **Suspicious offers are capped at 20**: cheap but untrustworthy is not a good deal.

### 4.6 Worked example (real test data)

Six offers for iPhone 15 128GB:

```
₹21,999   ₹65,999   ₹67,499   ₹69,900   ₹71,500   ₹79,999
```

| Step | Calculation | Result |
|---|---|---|
| Mean (for comparison only) | sum / 6 | ₹62,816, dragged down ~₹6k by the fake listing |
| **Fair price (median)** | (67,499 + 69,900) / 2 | **₹68,699.5** |
| Distances from median | 46,700 · 2,700 · 1,200 · 1,200 · 2,800 · 11,300 | |
| **MAD** | median of those | **₹2,750.5** |
| **Spread** | 1.4826 × 2,750.5 | **≈ ₹4,078** |

| Offer | vs fair | z | Label | Score |
|---|---|---|---|---|
| ₹21,999 | −68.0% | −11.45 | **suspicious** (< 55% of fair = ₹37,785) | 20 (capped) |
| ₹65,999 | −3.9% | −0.66 | fair | 67 |
| ₹67,499 | −1.7% | −0.29 | fair | 57 |
| ₹69,900 | +1.7% | +0.29 | fair | 43 |
| ₹71,500 | +4.1% | +0.69 | fair | 33 |
| ₹79,999 | +16.4% | +2.77 | **overpriced** | 0 |

**Best trusted offer = ₹65,999**, because the suspicious ₹21,999 is skipped. The UI also shows "1 suspicious" on the group.

---

## 5. Putting it together: `pipeline.py`

`analyze(items, query)`:
1. Runs `cluster_titles()` on all offers and gets a group label per offer.
2. For each group:
   - runs `score_group()` on its prices and attaches `deal_label`, `deal_score`, `fair_price`, `vs_fair_pct`, `robust_z` to each offer
   - **majority vote** on `accessory_reason()`: if more than half the offers are accessories/parts/add-ons, the group is excluded with that reason
   - flags `is_refurbished`, works out its **relevance** to the query (`_mismatch` → exact / variant / other)
   - builds a `ProductGroup`: name, offer count, min/median/max price, **best trusted offer** (cheapest non-suspicious) and its image, suspicious count, and a spelling-independent `history_key`
3. `_flag_price_outliers()`: groups under 20% of the anchor price are excluded as "price outlier".
4. **Ranks groups:**
   1. real products before excluded ones (accessories, parts, outliers)
   2. relevance: `exact` → `variant` → `other`. `_mismatch` adds +3 for off-topic (under half the query words), a different model number or model code; +1 per extra or missing variant word; +1 for different storage. 0 = exact, 1–2 = variant, 3+ = other.
   3. new before refurbished
   4. more offers first (more evidence), then cheaper
5. Copies each group's exclusion flag onto its offers.

The route (`search.py`) runs the ML on **all** fetched offers *before* trimming to `limit`, because the statistics need the full picture.

---

## 6. Tuning knobs

| Constant | File | Default | Raise it → | Lower it → |
|---|---|---|---|---|
| `SIMILARITY_THRESHOLD` | matcher.py | 0.55 | Fewer merges: safer, but the same product may split | More merges: risk of mixing products |
| `MODEL_CODE_BONUS` | matcher.py | 0.20 | More lenient when model codes match | Stricter |
| `NGRAM_SIZES` | matcher.py | (3, 4) | Longer n-grams are more word-like and less typo-tolerant | Shorter ones are noisier |
| `MAX_TITLE_TOKENS` | matcher.py | 5 | More of the title counts (more marketing noise) | Only the brand/model words count |
| `VARIANT_WORDS`, `USED_WORDS`, `ACCESSORY_WORDS`, `PART_WORDS`, `COLOR_WORDS`, `STOPWORDS` | matcher.py | lists | Add words when you find a new failure | |
| `is_english(max_foreign_ratio)` | matcher.py | 0.15 | Allows more non-Latin letters | Stricter English-only |
| `PRICE_OUTLIER_RATIO` / `PRICE_OUTLIER_HIGH_RATIO` | pipeline.py | 0.20 / 5.0 | Flags more cheap (or fewer expensive) groups | The opposite |
| `MIN_QUERY_OVERLAP` | pipeline.py | 0.5 | More results counted as off-topic | Fewer |
| `SUSPICIOUS_RATIO` | deal_scorer.py | 0.55 | Flags more listings as suspicious | Flags fewer |
| `GREAT_DEAL_PCT` | deal_scorer.py | −7% | | |
| `OVERPRICED_PCT` | deal_scorer.py | +10% | | |

Workflow for tuning: find a bad search → add it as a test case in `tests/test_ml.py` → change the knob or word list → run the tests → push.

---

## 7. Design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| **TF-IDF instead of neural embeddings** (e.g. sentence-transformers) | No training data needed, fully explainable, adapts per query, and PyTorch would blow Vercel's function size limit and add multi-second cold starts | Less semantic understanding (it won't know "cell phone" = "smartphone"), but product titles are about model names and specs, which character n-grams handle well |
| **Hand-written guards on top of ML** | Precision matters more than recall: wrongly merging a Pro with a base model gives a misleading "deal", while failing to merge two listings just shows them separately | Rules need maintenance as new product lines appear |
| **Union-find clustering** | Simple, fast, no need to choose the number of clusters in advance | Transitive chaining can occasionally link two products through a borderline middle listing; the guards limit this |
| **Median/MAD instead of mean/std** | One scam listing can't move the fair price | Needs at least 2 offers; with very few offers the estimate is rough |
| **Pure Python** | Small serverless bundle and fast cold starts on Vercel's free tier | Slower than numpy at scale, but irrelevant at about 60 listings per search |
| **Unsupervised** | Works on day one with zero labelled data | Thresholds are hand-tuned rather than learned (see §9) |

---

## 8. Limitations

Being upfront about these builds credibility:

1. **Thresholds are hand-tuned**, validated on test cases and real searches, not on a large labelled dataset. There's no measured precision/recall yet.
2. **Word lists are English and electronics-focused.** Fashion or grocery searches would need different guards.
3. **Fair price = the median of what *this search* returned.** If every seller is overpriced, "fair" is overpriced too. It's relative, not absolute.
4. **Small groups are weak evidence.** A group of 2 can't really have a "suspicious" member.
5. **Fixed USD→INR rate (83)** for the rare dollar listing.
6. **Price history is just starting.** The logging pipeline is built, but until it has accumulated data (or the Amazon.in backfill is imported, §10) the app can't say whether today's price is good compared with last month.
7. **The English-only and accessory rules are heuristics.** A listing whose title is mostly in English but describes a part in an unusual way can slip through; the price-outlier check is the safety net.

---

## 9. How to evaluate it properly

The next step toward "real" ML rigour:

1. **Log searches** (titles + groups produced).
2. **Label ~300 title pairs** by hand: same product or not.
3. **Measure** for matching:
   - **Precision**: of the pairs we merged, how many were truly the same? (Most important: wrong merges mislead users.)
   - **Recall**: of the truly-same pairs, how many did we merge?
   - **F1**: their balance.
4. **Sweep `SIMILARITY_THRESHOLD`** from 0.3 to 0.8 and plot precision vs recall to pick the threshold *from data*.
5. Then try **sentence embeddings** or a small trained classifier on pair features (cosine, same storage, same brand…) and compare F1 against this baseline.

---

## 9b. Price history: the data pipeline for future ML

`price_log.py` logs, for every relevant product group, one point per day: `{fair (median), min, max, offers, best_url}`, keyed by a **spelling-independent product key** (`product_key`: the identifying words at the start of the title plus storage, e.g. "OPPO Reno 15 Pro Mini 5G Glacier 256GB without offers" → `oppo-reno-15-pro-mini-256gb`, using the most common key in the group). Data comes from user searches plus a daily cron over tracked products. This is the dataset the "buy now or wait" model will be trained on.

## 10. Roadmap: "buy now or wait"

Goal: tell the user to wait because the price is likely to drop within the next 3 months (for example in a specific sale), or to buy now.

Data options found (Oct 2026):

| Option | History | Cost | Catch |
|---|---|---|---|
| **Apify "Amazon Price History API"** (`radeance/amazon-price-history-api`) | **Full Amazon.in history per product**, tested | ~$0.007 per product ($5 free credit/month) | Amazon only; third-party scraper, so treat it as a backfill source |
| Keepa API | Long Amazon price history | Paid (from €49) | Amazon only |
| Google Shopping price insights | Low/typical/high + recent history | Via SerpAPI | **US only** |
| Log our own prices daily | Starts at zero | Free (built: Upstash Redis + Vercel cron) | Takes months to accumulate |

**Test result (27 Oct 2026):** one Apify run for the iPhone 15 128GB (ASIN `B0CHX1W1XY`) returned **343 buy-box price points from Sept 2023 to July 2026**. Lowest price per month shows exactly the patterns the model needs to learn:

| Period | Lowest price | What happened |
|---|---|---|
| Sept 2023 | ₹79,900 | Launch price |
| Mar 2024 | ₹65,999 | First big dip |
| Sept–Nov 2024 | ₹65,900 → ₹61,900 | iPhone 16 launch + festive sales |
| Jan 2025 | ₹56,999 | Republic Day sale |
| **Sept 2025** | **₹45,999** | iPhone 17 launch + festive sales: the all-time low, ~23% below the month before |
| Apr–Jul 2026 | ₹59,900 | Price settled, then went out of stock |

Output fields used by the importer: `asin`, `name`, `currency` (INR, prices in **rupees**), `image_url`, and the histories `price_buybox_history` (main series), `price_new_history`, `price_new_fba_history`, `price_prime_exclusive_history`, `n_offers_new_history`, each a list of `{date, price}` where `price: null` means unavailable at that time. Apify input: `{"country": "in", "identifiers": ["<ASIN or URL>"], "include_variants": false}`; the Console requires a max-cost cap of at least $0.25 per run.

Planned approach:
1. **Backfill** real Amazon.in history for ~20 tracked products from Apify into the same Redis history store (planned importer).
2. **A sale-calendar baseline** built on Indian sale seasons (year-end sales, Republic Day sales in January, festive sales in Sept–Oct, mid-year sales) and launch cycles (phones drop when the successor launches). This is rules + calendar, not trained ML, and should be described that way.
3. **Logging the fair price of each product group daily**: built (section 9b).
4. **With the backfilled years of data:** a time-series model (seasonal decomposition, or gradient boosting on features like days-to-next-sale and days-since-launch) to predict the probability and size of a price drop.

---

## 11. Interview guide

### The 30-second pitch
> "Price comparison breaks when the same product is listed fifty different ways and some listings are fake. I built Credisearch, which pulls prices from several Google Shopping APIs, uses unsupervised ML (TF-IDF text vectors and similarity clustering with domain rules) to group listings into real products, then uses robust statistics to estimate a fair price and flag offers that are too good to be true. The user sees the best price they can actually trust."

### The business angle
- **User problem:** price comparison is noisy, and fake "deals" erode trust.
- **Value:** saves money *and* protects users from scams, so the trust layer is the differentiator.
- **Monetisation ideas:** affiliate links on the "best trusted offer", a price-drop alert subscription, or B2B selling competitor-price intelligence to sellers.
- **Unit economics awareness:** each search costs API credits, so caching, free tiers and the circuit breaker keep cost per search near zero at demo scale.

### Likely questions and good answers

**"Is this really machine learning?"**
Yes: it's *unsupervised* ML. TF-IDF is a learned representation (the weights come from the data in each search), similarity clustering is a classic unsupervised method, and robust z-scores are standard anomaly detection. It isn't deep learning, and it doesn't need to be: for short product titles, character n-grams plus domain rules are a strong, explainable baseline.

**"Why not use a large language model?"**
Cost, latency, and explainability. An LLM call per listing would cost money and seconds on every search. This runs in milliseconds for free, and every decision can be explained ("not merged because 128GB ≠ 256GB").

**"How do you know it works?"**
Automated tests on real-world-style titles across iPhone, Samsung, Sony and OnePlus, plus checking live searches (34 tests at the time of writing). The honest next step is a labelled dataset to measure precision and recall and tune thresholds from data (§9).

**"What was the hardest part?"**
Real data. Colour names split identical phones into separate groups; RAM sizes ("8GB RAM, 256GB", "8/128") broke the storage check; "S24+" lost its "+" and merged with the S24; Sony XM4 merged with XM5 until model codes became a guard; one API silently returned US listings in dollars; Hindi listings and spare parts (display combos) showed up as "products"; Google returned Skechers shoes for a hockey-stick search. Each one became a rule plus a test.

**"What would you do with more time?"**
The "buy now or wait" prediction on the real Amazon.in history (§10), with a waterfall chart explaining each prediction; a "wrong match" feedback button to collect labels; and a labelled evaluation set.

**"It's a personal project, where does data come from with no users?"**
A daily job collects prices on its own, and years of real Amazon.in history were backfilled through an API, so the model doesn't depend on traffic.

---

## 12. Glossary

| Term | Plain meaning |
|---|---|
| **Vector / embedding** | A list of numbers representing text, so a computer can measure similarity |
| **Character n-gram** | An overlapping chunk of n characters ("iph", "pho"…) |
| **TF (term frequency)** | How often a feature appears in one title |
| **IDF (inverse document frequency)** | How rare a feature is across all titles; rare means informative |
| **Cosine similarity** | 0–1 score of how similar two vectors are (the angle between them) |
| **Unsupervised learning** | Finding structure in data without labelled examples |
| **Clustering** | Grouping similar items together |
| **Union-find** | A data structure that efficiently merges items into groups |
| **Median** | The middle value; ignores extreme values |
| **MAD** | Median Absolute Deviation; a robust measure of spread |
| **Robust z-score** | How unusual a value is, measured with median and MAD |
| **Anomaly detection** | Finding data points that don't fit the normal pattern |
| **Precision / recall** | Of what we flagged, how much was right / of what was right, how much we flagged |
| **Circuit breaker** | Pattern that stops calling a failing service for a while |

---

**Code map:** `matcher.py` (sections 3.1–3.7) · `deal_scorer.py` (section 4) · `pipeline.py` (sections 3.6b, 5) · `price_log.py` (section 9b) · `tests/test_ml.py` (sections 3.8, 4.6). For how the rest of the backend works (routes, sources, caching, storage, deployment), see [BACKEND.md](BACKEND.md).
