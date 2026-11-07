"""Generate ShopFlow's fictional supplier invoices: PDFs in 3 layouts, noisy scans, ground truth.

Deterministic: the same --seed always produces byte-identical files (reportlab runs in
`invariant` mode, so no timestamps or random IDs leak into the PDFs).

    uv run python make_invoices.py --out data --seed 8

Layouts (each one a different way real suppliers print the same facts):
  A  US "classic": long dates (March 14, 2026), $1,234.56, SKU | Description | Qty | Price | Amount
  B  compact, US or UK supplier: 03/14/2026 vs 14/03/2026 (ambiguous!), SKU hidden under the
     description, "Amount due"
  C  German: Rechnung, 14.03.2026, 1.234,56 EUR, headings in German, delivery date as a distractor
Every layout also prints a due date, which is NOT the invoice date.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import numpy as np
from ch08_invoice import Invoice, LineItem
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
SYMBOL = {"USD": "$", "GBP": "£", "EUR": "€"}

# (name, address lines, country) per layout family. All fictional.
SUPPLIERS_US = [
    ("Harbor Lane Packaging Inc.", ["410 Quay Street", "Portland, OR 97209", "USA"]),
    ("Brightwave Textiles LLC", ["88 Mill Road", "Greensboro, NC 27401", "USA"]),
    ("Cobalt & Finch Hardware", ["1200 Lakeshore Ave", "Cleveland, OH 44113", "USA"]),
    ("Summit Ridge Outdoor Supply", ["35 Alpine Way", "Boulder, CO 80302", "USA"]),
    ("Maple Quay Stationery Co.", ["9 Canal Street", "Buffalo, NY 14202", "USA"]),
    ("Redwood Parcel Systems", ["2750 Harbor Blvd", "Oakland, CA 94607", "USA"]),
]
SUPPLIERS_UK = [
    ("Thistle & Oak Wholesale Ltd", ["14 Wharf Lane", "Leeds LS1 4AB", "United Kingdom"]),
    ("Kettleby Homeware Ltd", ["7 Trent Road", "Nottingham NG1 5FS", "United Kingdom"]),
    ("Pennine Cycle Parts Ltd", ["Unit 3, Don Works", "Sheffield S4 7YA", "United Kingdom"]),
]
SUPPLIERS_DE = [
    ("Nordlicht Verpackung GmbH", ["Speicherstraße 12", "20457 Hamburg", "Deutschland"]),
    ("Schwarzwald Werkzeuge AG", ["Bahnhofstraße 5", "79098 Freiburg", "Deutschland"]),
    ("Rheintal Textil GmbH", ["Hohenzollernring 40", "50672 Köln", "Deutschland"]),
]
ITEMS_EN = [
    ("Corrugated shipping box 40x30x20 cm", 85, 240), ("Packing tape 50 mm x 66 m", 120, 310),
    ("Bubble wrap roll 50 m", 1450, 2600), ("Cotton tote bag, natural", 180, 420),
    ("Stainless steel water bottle 750 ml", 650, 1190), ("Spiral notebook A5, 80 sheets", 140, 360),
    ("Ballpoint pens, box of 50", 480, 990), ("Bike inner tube 700x25c", 390, 720),
    ("LED desk lamp, black", 1900, 3400), ("Ceramic mug 350 ml", 210, 480),
    ("Wool throw blanket 130x170", 2400, 4800), ("Camping stove, single burner", 2100, 3900),
    ("Hiking socks, pair", 380, 760), ("Adhesive labels 100x150 mm, roll of 500", 560, 1200),
    ("Cable ties 300 mm, bag of 100", 190, 420), ("Garden gloves, size M", 260, 540),
    ("Reusable lunch box", 520, 980), ("Yoga mat 6 mm", 1100, 2200),
    ("Insulated shipping envelope", 95, 210), ("Pallet wrap 500 mm x 300 m", 1300, 2100),
    ("USB-C cable 1 m", 260, 640), ("Glass storage jar 1 L", 240, 530),
    ("Bamboo cutting board", 780, 1650), ("Rain poncho, adult", 230, 480),
]
ITEMS_DE = [
    ("Faltkarton 40x30x20 cm", 90, 260), ("Klebeband 50 mm x 66 m", 110, 300),
    ("Luftpolsterfolie Rolle 50 m", 1400, 2500), ("Baumwolltasche, natur", 170, 410),
    ("Edelstahl-Trinkflasche 750 ml", 640, 1150), ("Notizbuch A5, 80 Blatt", 130, 340),
    ("Kugelschreiber, Packung à 50", 470, 950), ("Fahrradschlauch 700x25c", 380, 700),
    ("LED-Schreibtischlampe, schwarz", 1800, 3300), ("Keramiktasse 350 ml", 200, 460),
    ("Wolldecke 130x170", 2300, 4700), ("Campingkocher, einflammig", 2000, 3800),
    ("Wandersocken, Paar", 360, 740), ("Etiketten 100x150 mm, Rolle à 500", 540, 1150),
    ("Kabelbinder 300 mm, Beutel à 100", 180, 400), ("Gartenhandschuhe, Größe M", 250, 520),
    ("Frischhaltedose", 500, 960), ("Yogamatte 6 mm", 1050, 2100),
    ("Thermo-Versandtasche", 90, 200), ("Stretchfolie 500 mm x 300 m", 1250, 2050),
]


@dataclass(frozen=True)
class InvoiceSpec:
    file: str
    layout: str  # "A" | "B" | "C"
    invoice: Invoice
    address: tuple[str, ...]
    date_style: str  # long | mdy | dmy | dotted
    money_style: str  # us | eu
    tax_bp: int  # tax rate in basis points (800 = 8%)
    issue: date
    due: date
    po: str


# ---------------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------------


def fmt_amount(cents: int, style: str) -> str:
    whole, frac = divmod(cents, 100)
    s = f"{whole:,}"
    return f"{s.replace(',', '.')},{frac:02d}" if style == "eu" else f"{s}.{frac:02d}"


def fmt_money(cents: int, spec: InvoiceSpec) -> str:
    amount = fmt_amount(cents, spec.money_style)
    sym = SYMBOL[spec.invoice.currency]
    return f"{amount} {sym}" if spec.money_style == "eu" else f"{sym}{amount}"


def fmt_date(d: date, style: str) -> str:
    if style == "long":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if style == "mdy":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == "dmy":
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    return f"{d.day:02d}.{d.month:02d}.{d.year}"


# ---------------------------------------------------------------------------------------------
# Specs (the ground truth)
# ---------------------------------------------------------------------------------------------


def _make_spec(rng: random.Random, idx: int, layout: str, n_lines: int) -> InvoiceSpec:
    if layout == "A":
        name, addr = rng.choice(SUPPLIERS_US)
        currency, date_style, money_style, tax_bp = "USD", "long", "us", rng.choice([0, 800, 888])
        inv_no = f"INV-2026-{rng.randint(100, 9999):05d}"
        catalog = ITEMS_EN
    elif layout == "B":
        if rng.random() < 0.5:
            name, addr = rng.choice(SUPPLIERS_UK)
            currency, date_style, tax_bp = "GBP", "dmy", 2000
        else:
            name, addr = rng.choice(SUPPLIERS_US)
            currency, date_style, tax_bp = "USD", "mdy", rng.choice([0, 800])
        money_style = "us"
        inv_no = f"{rng.choice('ABKT')}-{rng.randint(10000, 99999)}"
        catalog = ITEMS_EN
    else:
        name, addr = rng.choice(SUPPLIERS_DE)
        currency, date_style, money_style, tax_bp = "EUR", "dotted", "eu", 1900
        inv_no = f"RE-2026-{rng.randint(100, 9999):04d}"
        catalog = ITEMS_DE

    prefix = "".join(w[0] for w in name.replace("&", "").split()[:2]).upper()
    skus: set[str] = set()
    items: list[LineItem] = []
    for _ in range(n_lines):
        desc, lo, hi = rng.choice(catalog)
        sku = f"{prefix}-{rng.randint(1000, 9999)}"
        while sku in skus:
            sku = f"{prefix}-{rng.randint(1000, 9999)}"
        skus.add(sku)
        items.append(LineItem(sku=sku, desc=desc, qty=rng.choice([1, 2, 5, 10, 12, 24, 36, 48, 100]),
                              unit_price_cents=rng.randint(lo, hi)))
    subtotal = sum(li.qty * li.unit_price_cents for li in items)
    tax = (subtotal * tax_bp + 5000) // 10000  # round half up
    issue = date(2026, 1, 2) + timedelta(days=rng.randint(0, 300))
    inv = Invoice(supplier=name, invoice_no=inv_no, date=issue.isoformat(), currency=currency,
                  line_items=items, subtotal_cents=subtotal, tax_cents=tax,
                  total_cents=subtotal + tax)
    return InvoiceSpec(
        file=f"inv-{idx:03d}.pdf", layout=layout, invoice=inv, address=tuple(addr),
        date_style=date_style, money_style=money_style, tax_bp=tax_bp, issue=issue,
        due=issue + timedelta(days=rng.choice([14, 30, 45])), po=f"PO-{rng.randint(1000, 9999)}")


def generate_specs(seed: int = 8, n: int = 50) -> list[InvoiceSpec]:
    """`n` invoices, layouts evenly mixed, 3 of them long enough to spill onto a second page."""
    rng = random.Random(seed)
    layouts = (["A", "B", "C"] * (n // 3 + 1))[:n]
    rng.shuffle(layouts)
    long_ones = set(rng.sample(range(n), 3))
    specs = []
    for i, layout in enumerate(layouts):
        n_lines = rng.randint(38, 44) if i in long_ones else rng.randint(2, 9)
        specs.append(_make_spec(rng, i + 1, layout, n_lines))
    return specs


def pick_scan_sources(specs: list[InvoiceSpec], seed: int = 8, n_scans: int = 10) -> list[int]:
    """Indices of single-page invoices that also exist as a scan. Sorted, deterministic."""
    rng = random.Random(seed + 1)
    eligible = [i for i, s in enumerate(specs) if len(s.invoice.line_items) <= 12]
    return sorted(rng.sample(eligible, min(n_scans, len(eligible))))


# ---------------------------------------------------------------------------------------------
# PDF rendering (reportlab)
# ---------------------------------------------------------------------------------------------

_ss = getSampleStyleSheet()
_BODY = ParagraphStyle("body", parent=_ss["Normal"], fontName="Helvetica", fontSize=9, leading=11)
_SMALL = ParagraphStyle("small", parent=_BODY, fontSize=7, textColor=colors.grey, leading=8)
_RIGHT = ParagraphStyle("right", parent=_BODY, alignment=2)


def _p(text: str, style=_BODY) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def _totals(spec: InvoiceSpec, labels: tuple[str, str, str], width: float) -> Table:
    inv = spec.invoice
    rows = [[labels[0], fmt_money(inv.subtotal_cents, spec)],
            [labels[1], fmt_money(inv.tax_cents, spec)],
            [labels[2], fmt_money(inv.total_cents, spec)]]
    t = Table(rows, colWidths=[width * 0.6, width * 0.4], hAlign="RIGHT")
    t.setStyle(TableStyle([("ALIGN", (1, 0), (1, -1), "RIGHT"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
                           ("LINEABOVE", (0, 2), (-1, 2), 0.8, colors.black)]))
    return t


def _items_table(rows: list[list], widths: list[float], numeric_cols: range, head_bg) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), head_bg), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        *[("ALIGN", (c, 0), (c, -1), "RIGHT") for c in numeric_cols],
    ]))
    return t


def _story_a(spec: InvoiceSpec, width: float) -> list:
    inv, m = spec.invoice, lambda c: fmt_money(c, spec)
    head = Table([[[_p(f"<b><font size=15>{inv.supplier}</font></b>")]
                   + [_p(a) for a in spec.address],
                   _p("<b><font size=22>INVOICE</font></b>", _RIGHT)]],
                 colWidths=[width * 0.65, width * 0.35])
    meta = [_p(f"<b>Invoice No:</b> {inv.invoice_no}"),
            _p(f"<b>Invoice date:</b> {fmt_date(spec.issue, spec.date_style)}"),
            _p(f"<b>Due date:</b> {fmt_date(spec.due, spec.date_style)}"),
            _p(f"<b>Your PO:</b> {spec.po}")]
    bill = [_p("<b>Bill to</b>"), _p("ShopFlow Inc."), _p("Accounts Payable"),
            _p("350 Fifth Avenue, New York, NY 10118")]
    rows = [["SKU", "Description", "Qty", "Unit price", "Amount"]]
    rows += [[li.sku, _p(li.desc), str(li.qty), m(li.unit_price_cents),
              m(li.qty * li.unit_price_cents)] for li in inv.line_items]
    rate = f"{spec.tax_bp / 100:g}%"
    return [head, Spacer(1, 14), Table([[bill, meta]], colWidths=[width * 0.5, width * 0.5]),
            Spacer(1, 14),
            _items_table(rows, [width * w for w in (.16, .40, .08, .18, .18)], range(2, 5),
                         colors.Color(.9, .9, .9)),
            Spacer(1, 10), _totals(spec, ("Subtotal", f"Sales tax ({rate})", "Total"), width * .45),
            Spacer(1, 18), _p("Payment terms: net. Thank you for your business.", _SMALL)]


def _story_b(spec: InvoiceSpec, width: float) -> list:
    inv, m = spec.invoice, lambda c: fmt_money(c, spec)
    tax_label = "VAT" if inv.currency == "GBP" else "Sales tax"
    band = Table([[_p(f"<font color=white size=16><b>{inv.supplier}</b></font>")]],
                 colWidths=[width])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.Color(.15, .3, .5)),
                              ("TOPPADDING", (0, 0), (-1, -1), 10),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    left = [_p(a) for a in spec.address]
    right = [_p(f"Invoice <b>{inv.invoice_no}</b>", _RIGHT),
             _p(f"Issued {fmt_date(spec.issue, spec.date_style)}", _RIGHT),
             _p(f"Payment due {fmt_date(spec.due, spec.date_style)}", _RIGHT),
             _p(f"Order ref {spec.po}", _RIGHT)]
    rows = [["Item", "Qty", "Price", "Total"]]
    rows += [[_p(f"{li.desc}<br/><font size=7 color=grey>SKU: {li.sku}</font>"), str(li.qty),
              m(li.unit_price_cents), m(li.qty * li.unit_price_cents)] for li in inv.line_items]
    rate = f"{spec.tax_bp / 100:g}%"
    return [band, Spacer(1, 10), Table([[left, right]], colWidths=[width * .5, width * .5]),
            Spacer(1, 6), _p("Billed to: ShopFlow Inc., Accounts Payable, 350 Fifth Avenue, "
                             "New York, NY 10118"), Spacer(1, 14),
            _items_table(rows, [width * w for w in (.52, .1, .19, .19)], range(1, 4),
                         colors.Color(.85, .9, .96)),
            Spacer(1, 10), _totals(spec, ("Subtotal", f"{tax_label} {rate}", "Amount due"),
                                   width * .5)]


def _story_c(spec: InvoiceSpec, width: float) -> list:
    inv, m = spec.invoice, lambda c: fmt_money(c, spec)
    delivery = spec.issue - timedelta(days=2)
    head = Table([[[_p(f"<b>{inv.supplier}</b>")] + [_p(a) for a in spec.address],
                   _p("<b><font size=20>RECHNUNG</font></b>", _RIGHT)]],
                 colWidths=[width * .6, width * .4])
    meta = Table([["Rechnungsnr.:", inv.invoice_no], ["Datum:", fmt_date(spec.issue, "dotted")],
                  ["Lieferdatum:", fmt_date(delivery, "dotted")],
                  ["Zahlbar bis:", fmt_date(spec.due, "dotted")],
                  ["Bestellnr.:", spec.po]], colWidths=[width * .22, width * .3], hAlign="LEFT")
    meta.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9),
                              ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]))
    rows = [["Pos.", "Artikel-Nr.", "Bezeichnung", "Menge", "Einzelpreis", "Gesamt"]]
    rows += [[str(i), li.sku, _p(li.desc), str(li.qty), m(li.unit_price_cents),
              m(li.qty * li.unit_price_cents)] for i, li in enumerate(inv.line_items, 1)]
    return [head, Spacer(1, 14), _p("Rechnungsempfänger: ShopFlow Inc., Buchhaltung, "
                                    "350 Fifth Avenue, New York, NY 10118, USA"),
            Spacer(1, 10), meta, Spacer(1, 14),
            _items_table(rows, [width * w for w in (.07, .15, .33, .09, .18, .18)], range(3, 6),
                         colors.Color(.92, .92, .92)),
            Spacer(1, 10),
            _totals(spec, ("Zwischensumme", f"MwSt {spec.tax_bp / 100:g}%", "Gesamtbetrag"),
                    width * .5)]


def render_pdf(spec: InvoiceSpec, path: Path) -> None:
    page = LETTER if spec.layout == "A" else A4
    margin = 18 * mm
    doc = SimpleDocTemplate(str(path), pagesize=page, leftMargin=margin, rightMargin=margin,
                            topMargin=margin, bottomMargin=margin, invariant=1,
                            title=f"Invoice {spec.invoice.invoice_no}", author=spec.invoice.supplier)
    width = page[0] - 2 * margin
    story = {"A": _story_a, "B": _story_b, "C": _story_c}[spec.layout](spec, width)
    doc.build(story)


# ---------------------------------------------------------------------------------------------
# "Scans": draw the invoice with Pillow, then make it look like paper that went through a copier
# ---------------------------------------------------------------------------------------------

PAGE_PX = (1240, 1754)  # A4 at 150 dpi


def render_clean_image(spec: InvoiceSpec) -> Image.Image:
    inv, m = spec.invoice, lambda c: fmt_money(c, spec)
    de = spec.layout == "C"
    img = Image.new("L", PAGE_PX, 255)
    d = ImageDraw.Draw(img)
    font_path = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    big, mid, small = (ImageFont.truetype(font_path, s) for s in (42, 26, 22))
    x0, x1 = 90, PAGE_PX[0] - 90
    y = 90
    d.text((x0, y), inv.supplier, font=big, fill=0)
    d.text((x1, y), "RECHNUNG" if de else "INVOICE", font=big, fill=0, anchor="ra")
    y += 64
    for line in spec.address:
        d.text((x0, y), line, font=small, fill=40)
        y += 32
    y += 20
    labels = (
        [("Rechnungsnr.:", inv.invoice_no), ("Datum:", fmt_date(spec.issue, "dotted")),
         ("Zahlbar bis:", fmt_date(spec.due, "dotted")), ("Bestellnr.:", spec.po)] if de else
        [("Invoice No:", inv.invoice_no), ("Invoice date:", fmt_date(spec.issue, spec.date_style)),
         ("Due date:", fmt_date(spec.due, spec.date_style)), ("PO:", spec.po)])
    for k, v in labels:
        d.text((x0, y), k, font=mid, fill=0)
        d.text((x0 + 250, y), v, font=mid, fill=0)
        y += 40
    y += 30
    cols = {"sku": x0, "desc": x0 + 190, "qty": x1 - 330, "unit": x1 - 170, "amt": x1}
    heads = (["Artikel-Nr.", "Bezeichnung", "Menge", "Einzelpreis", "Gesamt"] if de
             else ["SKU", "Description", "Qty", "Unit price", "Amount"])
    d.line((x0, y - 6, x1, y - 6), fill=0, width=2)
    for key, h in zip(cols, heads, strict=True):
        d.text((cols[key], y), h, font=small, fill=0, anchor="la" if key in ("sku", "desc") else "ra")
    y += 38
    d.line((x0, y - 4, x1, y - 4), fill=0, width=2)
    for li in inv.line_items:
        d.text((cols["sku"], y), li.sku, font=small, fill=0)
        d.text((cols["desc"], y), li.desc[:34], font=small, fill=0)
        d.text((cols["qty"], y), str(li.qty), font=small, fill=0, anchor="ra")
        d.text((cols["unit"], y), m(li.unit_price_cents), font=small, fill=0, anchor="ra")
        d.text((cols["amt"], y), m(li.qty * li.unit_price_cents), font=small, fill=0, anchor="ra")
        y += 38
    y += 24
    tl = (("Zwischensumme", f"MwSt {spec.tax_bp / 100:g}%", "Gesamtbetrag") if de
          else ("Subtotal", f"Tax {spec.tax_bp / 100:g}%", "Total"))
    for label, cents in zip(tl, (inv.subtotal_cents, inv.tax_cents, inv.total_cents), strict=True):
        f = mid if label == tl[2] else small
        d.text((x1 - 420, y), label, font=f, fill=0)
        d.text((x1, y), m(cents), font=f, fill=0, anchor="ra")
        y += 40
    return img


def degrade(img: Image.Image, rng: np.random.Generator, *, angle: float, noise: float,
            blur: float, contrast: float) -> Image.Image:
    """Photocopier simulation: slightly gray paper, rotation, blur, grain, a few specks."""
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = img.point(lambda v: int(v * 0.93))  # off-white paper
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=220)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img, dtype=np.float32)
    arr += rng.normal(0, noise, arr.shape)
    specks = rng.random(arr.shape) < 0.0008
    arr[specks] = 40
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def render_scan(spec: InvoiceSpec, path: Path, seed: int, tier: str) -> None:
    rng = np.random.default_rng(seed)
    sign = 1 if rng.random() < 0.5 else -1
    if tier == "mild":
        params = {"angle": sign * rng.uniform(0.5, 1.5), "noise": 6.0, "blur": 0.6,
                  "contrast": 0.95}
    else:
        params = {"angle": sign * rng.uniform(3.5, 5.0), "noise": 14.0, "blur": 1.1,
                  "contrast": 0.8}
    degrade(render_clean_image(spec), rng, **params).save(path, format="PNG", compress_level=6)


# ---------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------


def write_dataset(out: Path, seed: int = 8, n: int = 50, n_scans: int = 10) -> dict:
    """Write PDFs, scans and ground_truth.json under `out`. Returns the ground-truth dict."""
    (out / "pdf").mkdir(parents=True, exist_ok=True)
    (out / "scans").mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*\n!.gitignore\n")
    specs = generate_specs(seed, n)
    files: dict[str, dict] = {}
    for spec in specs:
        render_pdf(spec, out / "pdf" / spec.file)
        files[spec.file] = {"kind": "pdf", "layout": spec.layout,
                            "invoice": spec.invoice.model_dump()}
    for k, idx in enumerate(pick_scan_sources(specs, seed, n_scans), start=1):
        spec = specs[idx]
        name = f"scan-{k:02d}.png"
        tier = "mild" if k <= n_scans - 4 else "hard"
        render_scan(spec, out / "scans" / name, seed * 1000 + k, tier)
        files[name] = {"kind": "scan", "layout": spec.layout, "tier": tier,
                       "source_pdf": spec.file, "invoice": spec.invoice.model_dump()}
    truth = {"seed": seed, "files": files}
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2, sort_keys=True) + "\n")
    return truth


def load_truth(data_dir: Path) -> dict[str, tuple[dict, Invoice]]:
    """file name -> (metadata, Invoice) from `ground_truth.json`."""
    raw = json.loads((data_dir / "ground_truth.json").read_text())["files"]
    return {name: (meta, Invoice.model_validate(meta["invoice"])) for name, meta in raw.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--scans", type=int, default=10)
    a = ap.parse_args()
    t = write_dataset(a.out, a.seed, a.n, a.scans)
    kinds = [v["kind"] for v in t["files"].values()]
    print(f"wrote {kinds.count('pdf')} PDFs and {kinds.count('scan')} scans to {a.out}/")
