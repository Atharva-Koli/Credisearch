"""Run from backend/:  python -m pytest tests -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.price import PriceItem
from app.services.ml import analyze
from app.services.ml.matcher import cluster_titles, is_accessory
from app.services.ml.deal_scorer import score_group


def test_same_product_different_wording_and_colour_merges():
    labels = cluster_titles([
        "Apple iPhone 15 (128 GB) - Black",
        "Apple iPhone 15 128GB Blue",
        "Apple iPhone 15, 128GB, Green",
    ])
    assert len(set(labels)) == 1


def test_variants_storage_and_generations_stay_separate():
    labels = cluster_titles([
        "Apple iPhone 15 128GB",
        "Apple iPhone 15 256GB",
        "Apple iPhone 15 Pro 128GB",
        "Apple iPhone 14 128GB",
    ])
    assert len(set(labels)) == 4


def test_model_codes_and_ram_handling():
    assert cluster_titles([
        "Sony WH-1000XM5 Wireless Headphones Black",
        "Sony WH1000XM5 Noise Cancelling Headphones Silver",
        "Sony WH-1000XM4 Wireless Headphones",
    ]) == [0, 0, 1]
    assert cluster_titles([
        "Samsung Galaxy S24 5G (8GB RAM, 256GB) Onyx Black",
        "Samsung Galaxy S24 256GB Marble Grey",
        "Samsung Galaxy S23 256GB",
    ]) == [0, 0, 1]


def test_accessory_detection_respects_query():
    assert is_accessory("Silicone Case for iPhone 15", "iphone 15")
    assert not is_accessory("Silicone Case for iPhone 15", "iphone 15 case")
    assert not is_accessory("Apple iPhone 15 128GB", "iphone 15")


def test_deal_labels():
    verdicts = score_group([21999, 65999, 67499, 69900, 71500, 79999])
    labels = [v["deal_label"] for v in verdicts]
    assert labels[0] == "suspicious"
    assert labels[-1] == "overpriced"
    assert set(labels[1:5]) == {"fair"}
    assert score_group([50000])[0]["deal_label"] == "no_comparison"


def test_pipeline_best_offer_skips_suspicious():
    items = [
        PriceItem(source="a", title="Apple iPhone 15 128GB Black", price=21999, currency="INR", url="fake"),
        PriceItem(source="a", title="Apple iPhone 15 128GB Blue", price=65999, currency="INR", url="real"),
        PriceItem(source="b", title="Apple iPhone 15 (128 GB) Green", price=69900, currency="INR", url="x"),
    ]
    enriched, groups = analyze(items, "iphone 15")
    assert len(groups) == 1
    assert groups[0].best_offer_url == "real"
    assert groups[0].suspicious_offers == 1
    assert all(it.group_id == groups[0].group_id for it in enriched)


def test_refurbished_kept_separate_from_new():
    labels = cluster_titles([
        "Apple iPhone 15 128GB Black",
        "Refurbished Apple iPhone 15 128GB Black by Cashify",
        "Apple iPhone 15 128GB - Renewed",
    ])
    assert labels[0] not in labels[1:]


def test_display_name_brackets():
    from app.services.ml.matcher import display_name
    assert display_name("Apple iPhone 15 (128 GB, Black)") == "Apple iPhone 15 (128 GB)"
    assert display_name("Apple iPhone 15 (128 GB) - Black") == "Apple iPhone 15 (128 GB)"


def test_circuit_breaker_disables_source_on_quota_error():
    import httpx
    from app.services import source_health as sh
    req = httpx.Request("GET", "https://x")
    quota = httpx.HTTPStatusError("quota", request=req, response=httpx.Response(429, request=req))
    sh.record_failure("demo", quota)
    assert not sh.is_available("demo")
    assert sh.disabled_sources()["demo"] == "HTTP 429"
    # a timeout / 500 must NOT disable a source
    sh.record_failure("other", httpx.ReadTimeout("slow"))
    server = httpx.HTTPStatusError("boom", request=req, response=httpx.Response(500, request=req))
    sh.record_failure("other", server)
    assert sh.is_available("other")


def test_circuit_breaker_on_error_message():
    from app.services import source_health as sh
    sh.record_failure("msg", RuntimeError("searchapi error: You have run out of credits"))
    assert not sh.is_available("msg")
    sh.record_failure("cfg", RuntimeError("SERPER_API_KEY is not configured"))
    assert sh.is_available("cfg")


def test_plus_sign_is_a_variant():
    assert cluster_titles([
        "Samsung Galaxy S24+ 256GB",
        "Samsung Galaxy S24 Plus 256GB Onyx Black",
        "Samsung Galaxy S24 256GB",
    ]) == [0, 0, 1]


def test_ram_storage_shorthand():
    from app.services.ml.matcher import extract_attributes
    assert extract_attributes("Samsung Galaxy S24 8/128")["storage"] == {"128gb"}
    assert extract_attributes("Samsung Galaxy S24 (8GB/256GB)")["storage"] == {"256gb"}
    assert cluster_titles(["Samsung Galaxy S24 8/128", "Samsung Galaxy S24 8/256"]) == [0, 1]


def test_ranking_puts_searched_model_first():
    titles = ["Samsung Galaxy S23 256GB", "Samsung Galaxy S23 256 GB Black",
              "Samsung Galaxy S26 Ultra 256GB", "Samsung Galaxy S24 256GB"]
    items = [PriceItem(source="s", title=t, price=50000 + i, currency="INR", url="u")
             for i, t in enumerate(titles)]
    _, groups = analyze(items, "samsung galaxy s24 256gb")
    assert "S24" in groups[0].name


# ---------------- outlier elimination (Oppo Reno screenshot cases) ----------------

def test_non_english_titles_detected():
    from app.services.ml.matcher import is_english
    assert is_english("OPPO Reno15 Pro 5G (12GB RAM, 256GB)")
    assert is_english("Café Edition Phone")
    assert not is_english("Oppo Reno 15 PRO MINI 5G स्किन नॉट बैक कवर, कैमरा और बैक प्रोटेक्टर मोबाइल स्किन")


def test_spare_parts_and_addons_flagged():
    from app.services.ml.matcher import accessory_reason
    q = "oppo reno 15 pro"
    assert accessory_reason("Display for Oppo Reno15 Pro Mini - (Combo Folder) | Maxbhi", q) == "spare part"
    assert accessory_reason("Oppo Reno 15 Pro Mini Display and Touchscreen Combo - 100% Original", q) == "spare part"
    assert accessory_reason("Battery Door for Oppo Reno 15 Pro", q) == "add-on"
    assert accessory_reason("OPPO Reno15 Pro 5G (12GB RAM, 256GB)", q) is None
    assert accessory_reason("Apple iPhone 15 for Rs 59,999", "iphone 15") is None


def test_price_outlier_and_other_models():
    titles_prices = [
        ("OPPO Reno15 Pro 5G 256GB", 60859), ("OPPO Reno 15 Pro 5G (256 GB)", 67999),
        ("Oppo Reno 15 Pro 5G 256GB Silver", 64999),
        ("Oppo Reno 15 Pro Mini 5G 256GB", 49999),          # a variant
        ("Oppo Reno 14 Pro 5G 256GB", 44999),               # a different model
        ("Mystery item Oppo Reno 15 Pro shell", 899),       # slipped past word lists
    ]
    items = [PriceItem(source="s", title=t, price=p, currency="INR", url=f"u{i}")
             for i, (t, p) in enumerate(titles_prices)]
    _, groups = analyze(items, "oppo reno 15 pro")
    by = {g.name: g for g in groups}
    main = groups[0]
    assert "15 Pro" in main.name and "Mini" not in main.name and main.offers == 3
    mini = next(g for g in groups if "Mini" in g.name)
    assert mini.relevance == "variant"
    r14 = next(g for g in groups if "14" in g.name)
    assert r14.relevance == "other"
    shell = next(g for g in groups if "shell" in g.name)
    assert shell.is_accessory and shell.excluded_reason == "price outlier"
    assert groups[-1] is shell  # excluded groups rank last


# ---------------- price logger with an in-memory fake Redis ----------------

class FakeRedis:
    def __init__(self):
        self.h, self.l = {}, {}

    def __call__(self, commands):
        out = []
        for cmd in commands:
            op, key, *args = cmd
            if op == "LPUSH":
                self.l.setdefault(key, []).insert(0, args[0]); out.append(len(self.l[key]))
            elif op == "LTRIM":
                self.l[key] = self.l.get(key, [])[args[0]:args[1] + 1]; out.append("OK")
            elif op == "LRANGE":
                out.append(self.l.get(key, [])[args[0]:args[1] + 1])
            elif op == "HSET":
                self.h.setdefault(key, {})[args[0]] = args[1]; out.append(1)
            elif op == "HGET":
                out.append(self.h.get(key, {}).get(args[0]))
            elif op == "HGETALL":
                out.append([x for kv in self.h.get(key, {}).items() for x in kv])
        return out


def test_price_logger_records_daily_points(monkeypatch):
    from app.services import price_log
    fake = FakeRedis()
    monkeypatch.setattr(price_log, "_pipeline", fake)
    monkeypatch.setattr(price_log, "enabled", lambda: True)

    items = [PriceItem(source="s", title=t, price=p, currency="INR", url="u")
             for t, p in [("Apple iPhone 15 128GB", 65999), ("Apple iPhone 15 (128 GB) Blue", 69900),
                          ("Silicone Case for iPhone 15", 499)]]
    _, groups = analyze(items, "iphone 15")
    price_log.log_search("iphone 15", groups, ["serpapi"])

    products = price_log.list_products()
    assert len(products) == 1                       # the case is not logged
    key = next(iter(products))
    hist = price_log.get_history(key)
    assert len(hist["points"]) == 1 and hist["points"][0]["fair"] == 67949.5
    assert price_log.recent_events(5)[0]["q"] == "iphone 15"

    # simulate yesterday's point -> history grows to 2 days
    fake.h[f"price:{key}"]["2026-01-01"] = '{"fair": 70000, "min": 69000, "max": 71000, "offers": 2, "best_url": "u"}'
    assert [p["date"] for p in price_log.get_history(key)["points"]][0] == "2026-01-01"


def test_glued_model_number_compared_with_spaced_one():
    assert cluster_titles([
        "OPPO Reno15 Pro 5G 256GB", "Oppo Reno 14 Pro 5G 256GB", "OPPO Reno 15 Pro 5G 256 GB",
    ]) == [0, 1, 0]


def test_history_key_is_spelling_independent():
    from app.services.ml.matcher import canonical_key
    keys = {canonical_key(t) for t in [
        "OPPO Reno15 Pro 5G (12GB RAM, 256GB) Sunset Gold",
        "OPPO Reno 15 Pro 5G 256 GB Titanium Grey",
        "Oppo Reno15 Pro 256GB",
    ]}
    assert keys == {"oppo-reno-15-pro-256gb"}


def test_real_oppo_listings():
    assert cluster_titles([
        "OPPO Reno15 Pro 5G 512GB",
        "OPPO Reno15 Pro 5G - 200MP Camera & Dimensity 8450 / 12GB 512GB",
        "OPPO Reno15 Pro 5G Smartphone 12+512GB | 6.7-inch 120Hz",
        "Oppo Reno15c 5G",
        "oppo Reno15 5G",
    ]) == [0, 0, 0, 1, 2]


def test_missing_variant_is_not_exact():
    items = [PriceItem(source="s", title=t, price=p, currency="INR", url="u") for t, p in
             [("OPPO Reno15 Pro 5G 256GB", 64999), ("oppo Reno15 5G 256GB", 52999)]]
    _, groups = analyze(items, "oppo reno 15 pro")
    rel = {g.name: g.relevance for g in groups}
    assert groups[0].name.startswith("OPPO Reno15 Pro")
    assert rel[groups[1].name] == "variant"


def test_product_never_merges_with_its_part():
    assert cluster_titles([
        "Oppo Reno 15 Pro Mini 5G 256GB",
        "Oppo Reno 15 Pro Mini Display and Touchscreen Combo - 100% Original",
        "Oppo Reno 15 Pro Mini 5G (256 GB) Silver",
    ]) == [0, 1, 0]


def test_off_topic_results_are_other():
    items = [PriceItem(source="s", title=t, price=p, currency="INR", url="u") for t, p in [
        ("SKECHERS Mens DYNAMITE AT Casual Sneakers Shoe", 2999), ("SKECHERS Mens DYNAMITE AT Sneakers", 3419),
        ("ALFA Y-30 Junior Composite Hockey Stick", 1890), ("Hama Star 78 Mono Monopod", 2399)]]
    _, groups = analyze(items, "alfa hockey stick")
    assert groups[0].name.startswith("ALFA")
    assert {g.relevance for g in groups[1:]} == {"other"}
    # normal searches are unaffected
    items = [PriceItem(source="s", title="Apple iPhone 15 (128 GB) - Black", price=69900, currency="INR", url="u")]
    assert analyze(items, "iphone 15 128gb")[1][0].relevance == "exact"


def test_decimal_screen_size_and_high_price_outlier():
    assert cluster_titles([
        "Apple iPhone 15 6,1 128GB", "Apple iPhone 15 128GB", "Apple iPhone 14 6,1 128GB",
    ]) == [0, 0, 1]
    from app.models.price import ProductGroup
    from app.services.ml.pipeline import _flag_price_outliers

    def grp(gid, offers, med):
        return ProductGroup(group_id=gid, name=gid, offers=offers, min_price=med, median_price=med,
                            max_price=med, best_offer_url="u", best_offer_price=med, is_accessory=False,
                            suspicious_offers=0)

    groups = [grp("iphone", 6, 66999), grp("tiktok", 2, 1785566), grp("skin", 1, 229)]
    _flag_price_outliers(groups)
    reasons = {g.group_id: g.excluded_reason for g in groups}
    assert reasons == {"iphone": None, "tiktok": "price outlier", "skin": "price outlier"}


def test_mojibake_hindi_is_repaired_then_filtered():
    from app.services.aggregator import fix_mojibake
    from app.services.ml.matcher import is_english
    garbled = "एप्पल iPhone 17".encode("utf-8").decode("latin-1")
    assert is_english(garbled)                      # the bug: looks Latin
    assert fix_mojibake(garbled) == "एप्पल iPhone 17"
    assert not is_english(fix_mojibake(garbled))    # now correctly dropped
    assert fix_mojibake("Apple iPhone 17 (Sage, 256 GB)") == "Apple iPhone 17 (Sage, 256 GB)"


def test_logger_uses_trusted_price_and_dedupes_keys(monkeypatch):
    from app.services import price_log
    fake = FakeRedis()
    monkeypatch.setattr(price_log, "_pipeline", fake)
    monkeypatch.setattr(price_log, "enabled", lambda: True)
    items = [PriceItem(source="s", title=t, price=p, currency="INR", url="u") for t, p in [
        ("Apple iPhone 15 128GB", 21999), ("Apple iPhone 15 128GB Blue", 65999),
        ("Apple iPhone 15 (128 GB) Green", 67499), ("Apple iPhone 15 128GB Pink", 69900)]]
    _, groups = analyze(items, "iphone 15")
    price_log.log_search("iphone 15", groups, ["s"])
    key = next(iter(price_log.list_products()))
    point = price_log.get_history(key)["points"][0]
    assert point["min"] == 65999          # the suspicious ₹21,999 is not the "lowest price"
