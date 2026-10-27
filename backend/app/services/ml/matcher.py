"""
Product matching: TF-IDF character n-gram embeddings + cosine similarity
+ threshold clustering (union-find), with hard attribute guards.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set

SIMILARITY_THRESHOLD = 0.55
MODEL_CODE_BONUS = 0.2
NGRAM_SIZES = (3, 4)

# Words that change WHICH product it is (never merge across these)
VARIANT_WORDS = {
    "pro", "max", "plus", "mini", "ultra", "lite", "se", "fe", "air",
    "neo", "edge", "fold", "flip",
}

# Condition words: a refurbished phone is not the same offer as a new one
USED_WORDS = {
    "refurbished", "renewed", "used", "preowned", "pre-owned", "secondhand",
    "unboxed", "open-box", "openbox", "parts", "cashify", "ovantica",
}

# Words that mean the listing is an accessory, not the product itself
ACCESSORY_WORDS = {
    "case", "cover", "covers", "cases", "protector", "tempered", "glass",
    "screenguard", "guard", "skin", "cable", "charger", "adapter",
    "holder", "stand", "mount", "strap", "pouch", "sleeve", "lens",
    "sticker", "back", "film", "dock", "replacement",
}

# Marketing / noise words that carry no product identity
STOPWORDS = {
    "the", "with", "and", "for", "new", "latest", "buy", "online", "best",
    "price", "in", "india", "of", "a", "an", "by", "brand", "original",
    "genuine", "free", "delivery", "-", "|", "sealed", "edition",
    "5g", "4g", "lte", "ram", "rom", "storage", "smartphone", "mobile",
}

# Colours/finishes: same product for price comparison, so ignore them
COLOR_WORDS = {
    "black", "white", "blue", "green", "pink", "yellow", "red", "purple",
    "silver", "gold", "grey", "gray", "graphite", "midnight", "starlight",
    "titanium", "natural", "desert", "cream", "violet", "lavender", "mint",
    "orange", "teal", "navy", "beige", "bronze", "rose", "space", "jet",
    "phantom", "onyx", "obsidian", "porcelain", "hazel", "coral", "lime",
    "marble", "cobalt", "amber", "sapphire", "emerald", "charcoal",
}

_CONNECTIVITY = {"5g", "4g", "3g"}

_STORAGE_RE = re.compile(r"(\d+)\s*(gb|tb)\b")
_BARE_NUM_RE = re.compile(r"\b(\d{1,4})\b(?!\s*(?:gb|tb|mah|mp|hz|inch|in|w|cm|mm))")


def normalize(title: str) -> str:
    t = title.lower()
    t = re.sub(r"(?<=[a-z0-9])\+", " plus", t)  # "s24+" -> "s24 plus" (a different product)
    t = re.sub(r"[()\[\],/|+_]", " ", t)
    t = re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", "", t)  # "wh-1000xm5" -> "wh1000xm5"
    t = re.sub(r"(\d)\s+(gb|tb)\b", r"\1\2", t)  # "128 gb" -> "128gb"
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokens(title: str) -> List[str]:
    return [w for w in normalize(title).split() if w not in STOPWORDS and w not in COLOR_WORDS]


def extract_attributes(title: str) -> Dict[str, Set[str]]:
    t = normalize(title)
    words = set(t.split())
    sizes = [(int(n) * (1024 if u == "tb" else 1), f"{n}{u}") for n, u in _STORAGE_RE.findall(t)]
    # when a title lists RAM and storage ("8GB RAM, 256GB"), the largest is storage
    storage = {max(sizes)[1]} if sizes else set()
    size_tokens = {s for _, s in sizes}
    # bare numbers like the "15" in "iphone 15" (model generation)
    model_nums = set(_BARE_NUM_RE.findall(t))
    # alphanumeric model codes like "s24", "wh1000xm5", "a55"
    model_codes = {
        w for w in words
        if re.search(r"[a-z]", w) and re.search(r"\d", w)
        and w not in size_tokens and w not in _CONNECTIVITY
        and not re.fullmatch(r"\d+(gb|tb|mah|mp|hz|w|mm|cm|inch)", w)
    }
    return {
        "storage": storage,
        "variants": words & VARIANT_WORDS,
        "used": bool(words & USED_WORDS) or "pre owned" in t or "open box" in t,
        "model_nums": model_nums,
        "model_codes": model_codes,
    }


def display_name(title: str) -> str:
    """Title with colour words removed, for naming a product group."""
    words = [w for w in title.split() if w.strip("()[],-|").lower() not in COLOR_WORDS]
    name = " ".join(words)
    name = re.sub(r"\(\s*\)", "", name)
    name = re.sub(r"[\s,\-|/]+(?=\)|$)", "", name)
    # rebalance brackets left broken by removing a colour word
    while name.count(")") > name.count("("):
        i = name.rfind(")")
        name = name[:i] + name[i + 1:]
    name += ")" * (name.count("(") - name.count(")"))
    return re.sub(r"\s+", " ", name).strip() or title


def is_used(title: str) -> bool:
    return extract_attributes(title)["used"]


def is_accessory(title: str, query: str) -> bool:
    title_words = set(normalize(title).split())
    query_words = set(normalize(query).split())
    hits = (title_words & ACCESSORY_WORDS) - query_words
    return bool(hits)


def _char_ngrams(text: str) -> List[str]:
    padded = f" {text} "
    grams = []
    for n in NGRAM_SIZES:
        grams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
    return grams


def tfidf_vectors(titles: List[str]) -> List[Dict[str, float]]:
    """Embed each title as an L2-normalised TF-IDF vector over char n-grams."""
    docs = [Counter(_char_ngrams(" ".join(tokens(t)))) for t in titles]
    n_docs = len(docs)
    df = Counter()
    for d in docs:
        df.update(d.keys())
    idf = {g: math.log((1 + n_docs) / (1 + c)) + 1.0 for g, c in df.items()}

    vectors = []
    for d in docs:
        v = {g: (1 + math.log(tf)) * idf[g] for g, tf in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vectors.append({g: x / norm for g, x in v.items()})
    return vectors


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(g, 0.0) for g, x in a.items())


def compatible(a: Dict[str, Set[str]], b: Dict[str, Set[str]]) -> bool:
    """Hard guards: two listings can only be the same product if these agree."""
    if a["variants"] != b["variants"]:
        return False
    if a["used"] != b["used"]:
        return False
    if a["storage"] and b["storage"] and a["storage"] != b["storage"]:
        return False
    if a["model_nums"] and b["model_nums"] and not (a["model_nums"] & b["model_nums"]):
        return False
    if a["model_codes"] and b["model_codes"] and not (a["model_codes"] & b["model_codes"]):
        return False
    return True


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_titles(titles: List[str], threshold: Optional[float] = None) -> List[int]:
    """Return a cluster label per title (labels are 0..k-1, ordered by first appearance)."""
    threshold = SIMILARITY_THRESHOLD if threshold is None else threshold
    n = len(titles)
    if n == 0:
        return []

    vecs = tfidf_vectors(titles)
    attrs = [extract_attributes(t) for t in titles]
    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if not compatible(attrs[i], attrs[j]):
                continue
            # a shared model code (e.g. "wh1000xm5") is strong evidence of identity,
            # so accept looser wording around it
            pair_threshold = threshold
            if attrs[i]["model_codes"] & attrs[j]["model_codes"]:
                pair_threshold -= MODEL_CODE_BONUS
            if cosine(vecs[i], vecs[j]) >= pair_threshold:
                uf.union(i, j)

    roots: Dict[int, int] = {}
    labels = []
    for i in range(n):
        r = uf.find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels.append(roots[r])
    return labels
