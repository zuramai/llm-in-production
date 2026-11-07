import sys
from collections import Counter
from pathlib import Path

import pytest
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date

from ch08_extract import has_text_layer, pdf_text
from ch08_invoice import validate_invoice
from ch08_pipeline import route
from make_invoices import (
    fmt_amount,
    fmt_date,
    generate_specs,
    load_truth,
    pick_scan_sources,
    write_dataset,
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("ch08data")
    write_dataset(out, seed=8)
    return out


def test_counts_and_layout_mix(dataset):
    assert len(list((dataset / "pdf").glob("*.pdf"))) == 50
    assert len(list((dataset / "scans").glob("*.png"))) == 10
    truth = load_truth(dataset)
    layouts = Counter(m["layout"] for n, (m, _) in truth.items() if m["kind"] == "pdf")
    assert set(layouts) == {"A", "B", "C"}
    assert min(layouts.values()) >= 14


def test_same_seed_is_byte_identical(dataset, tmp_path):
    write_dataset(tmp_path, seed=8)
    for rel in ["ground_truth.json", "pdf/inv-001.pdf", "pdf/inv-050.pdf", "scans/scan-01.png",
                "scans/scan-10.png"]:
        assert (dataset / rel).read_bytes() == (tmp_path / rel).read_bytes(), rel


def test_different_seed_changes_data():
    a = [s.invoice.invoice_no for s in generate_specs(seed=8)]
    b = [s.invoice.invoice_no for s in generate_specs(seed=9)]
    assert a != b


def test_ground_truth_is_internally_consistent():
    # Generated truth must itself pass the validator, or the evals would punish correct answers.
    for spec in generate_specs(seed=8):
        assert validate_invoice(spec.invoice, today=date(2026, 11, 7)) == [], spec.file


def test_scans_point_at_single_page_invoices(dataset):
    truth = load_truth(dataset)
    for name, (meta, inv) in truth.items():  # noqa: PERF102
        if meta["kind"] == "scan":
            assert len(inv.line_items) <= 12
            assert truth[meta["source_pdf"]][1] == inv


def test_scan_sources_are_unique_and_sorted():
    idx = pick_scan_sources(generate_specs(seed=8), seed=8)
    assert idx == sorted(set(idx)) and len(idx) == 10


def test_pdfs_have_a_text_layer_and_some_span_two_pages(dataset):
    pages = [len(PdfReader(str(p)).pages) for p in sorted((dataset / "pdf").glob("*.pdf"))]
    assert pages.count(2) == 3 and max(pages) == 2
    assert all(has_text_layer(pdf_text(p)) for p in (dataset / "pdf").glob("*.pdf"))


def test_router_sends_pdf_to_text_and_png_to_vision(dataset):
    assert route(dataset / "pdf" / "inv-001.pdf")[0] == "text"
    assert route(dataset / "scans" / "scan-01.png")[0] == "vision"


def test_printed_text_contains_the_truth_numbers(dataset):
    # The text layer must actually carry the figures, in each layout's own number format.
    truth = load_truth(dataset)
    for name in ["inv-001.pdf", "inv-002.pdf", "inv-003.pdf"]:
        meta, inv = truth[name]
        text = pdf_text(dataset / "pdf" / name)
        style = "eu" if meta["layout"] == "C" else "us"
        assert inv.invoice_no in text
        assert fmt_amount(inv.total_cents, style) in text


@pytest.mark.parametrize(("cents", "style", "want"), [
    (123456, "us", "1,234.56"), (123456, "eu", "1.234,56"), (5, "us", "0.05"),
    (100000000, "eu", "1.000.000,00"),
])
def test_amount_formats(cents, style, want):
    assert fmt_amount(cents, style) == want


def test_ambiguous_date_styles_really_are_ambiguous():
    d = date(2026, 3, 4)
    assert fmt_date(d, "mdy") == "03/04/2026" and fmt_date(d, "dmy") == "04/03/2026"
    assert fmt_date(d, "dotted") == "04.03.2026" and fmt_date(d, "long") == "March 4, 2026"
