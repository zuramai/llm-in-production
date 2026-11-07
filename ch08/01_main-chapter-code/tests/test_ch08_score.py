import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch08_extract import STRICT_FORMAT
from ch08_invoice import Invoice, LineItem, extract_json, score_invoice, strict_schema


def truth() -> Invoice:
    return Invoice(
        supplier="Nordlicht Verpackung GmbH", invoice_no="RE-2026-1475", date="2026-04-02",
        currency="EUR",
        line_items=[LineItem(sku="NV-7000", desc="Flasche", qty=2, unit_price_cents=1057),
                    LineItem(sku="NV-1607", desc="Schlauch", qty=24, unit_price_cents=493)],
        subtotal_cents=13946, tax_cents=2650, total_cents=16596)


def test_perfect_prediction_scores_all_ones():
    s = score_invoice(truth(), truth())
    assert s["all"] == 1.0 and s["line_item_recall"] == 1.0


def test_failed_extraction_scores_zero_everywhere():
    s = score_invoice(None, truth())
    assert s["all"] == 0.0 and sum(s.values()) == 0.0


def test_swapped_day_and_month_only_breaks_date():
    p = truth()
    p.date = "2026-02-04"
    s = score_invoice(p, truth())
    assert s["date"] == 0.0 and s["total_cents"] == 1.0 and s["all"] == 0.0


def test_one_misread_digit_in_one_line_gives_partial_recall():
    p = truth()
    p.line_items[1].unit_price_cents = 498  # 493 misread as 498
    s = score_invoice(p, truth())
    assert s["line_item_recall"] == 0.5 and s["line_items"] == 0.0


def test_line_order_does_not_matter_but_extras_do():
    p = truth()
    p.line_items.reverse()
    assert score_invoice(p, truth())["line_items"] == 1.0
    p.line_items.append(LineItem(sku="X-1", desc="ghost", qty=1, unit_price_cents=1))
    s = score_invoice(p, truth())
    assert s["line_items"] == 0.0 and s["line_item_recall"] == 1.0


def test_supplier_and_sku_comparison_ignores_case_punctuation_spacing():
    p = truth()
    p.supplier = "NORDLICHT  Verpackung GmbH."
    p.line_items[0].sku = "nv - 7000"
    s = score_invoice(p, truth())
    assert s["supplier"] == 1.0 and s["line_items"] == 1.0


def test_duplicate_lines_are_matched_one_to_one():
    t = truth()
    t.line_items = [t.line_items[0], t.line_items[0].model_copy()]
    p = t.model_copy(deep=True)
    p.line_items = [t.line_items[0]]
    assert score_invoice(p, t)["line_item_recall"] == 0.5


def test_strict_schema_is_closed_inlined_and_fully_required():
    schema = strict_schema(Invoice)
    assert "$defs" not in str(schema) and "$ref" not in str(schema)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    assert schema["properties"]["line_items"]["items"]["properties"]["qty"]["type"] == "integer"
    assert STRICT_FORMAT["json_schema"]["strict"] is True


@pytest.mark.parametrize("raw", ['```json\n{"a": 1}\n```', 'Sure! {"a": 1} Hope that helps.',
                                 '{"a": 1}'])
def test_extract_json_finds_outermost_object(raw):
    assert extract_json(raw) == {"a": 1}


def test_extract_json_rejects_text_without_object():
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("I could not read this document.")
