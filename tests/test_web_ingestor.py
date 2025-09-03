import re
import sys
from pathlib import Path

# Ensure the repository root is on the import path for direct module imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from web_ingestor import parse_product_page, classify_by_thresholds, find_links


def test_classify_by_thresholds_selects_correct_class():
    thresholds = [
        {"kg_min": 5, "clase": "XL"},
        {"kg_min": 0, "clase": "XS"},
        {"kg_min": 1, "clase": "S"},
    ]
    assert classify_by_thresholds(2, thresholds) == "S"
    assert classify_by_thresholds(None, thresholds) == ""


def test_parse_product_page_jsonld_dimensions_and_classification():
    html = """
    <html><head><title>dummy</title>
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Product",
      "name": "Test Widget",
      "brand": {"name": "WidgetCo"},
      "weight": {"value": "1.5", "unitCode": "KGM"},
      "additionalProperty": [
        {"name": "Largo", "value": "20"},
        {"name": "Ancho", "value": "10"},
        {"name": "Alto", "value": "5"}
      ]
    }
    </script>
    </head><body></body></html>
    """
    thresholds = [
        {"kg_min": 0, "clase": "XS"},
        {"kg_min": 1, "clase": "M"},
        {"kg_min": 2, "clase": "L"},
    ]
    row = parse_product_page(html, "https://example.com/product", thresholds, divisor_vol=5000)

    assert row["product_name"] == "Test Widget"
    assert row["brand"] == "WidgetCo"
    assert row["peso_kg"] == 1.5
    assert row["largo_cm"] == 20.0
    assert row["ancho_cm"] == 10.0
    assert row["alto_cm"] == 5.0
    assert row["peso_vol_kg"] == 0.2
    assert row["peso_fact_kg"] == 1.5
    assert row["clase_logistica"] == "M"
    assert re.fullmatch(r"[0-9a-f]{64}", row["hash_row"])


def test_find_links_cross_domain_option():
    html = (
        '<a href="http://example.com/a">A</a>'
        '<a href="http://other.com/b">B</a>'
    )
    base = "http://example.com"
    same = find_links(html, base, same_domain_only=True)
    assert "http://example.com/a" in same
    assert "http://other.com/b" not in same
    wide = find_links(html, base, same_domain_only=False)
    assert "http://example.com/a" in wide
    assert "http://other.com/b" in wide
