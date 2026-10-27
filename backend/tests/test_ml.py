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
