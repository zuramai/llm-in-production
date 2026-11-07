"""Three ways to turn a document into an `Invoice`: text layer, vision, and OCR-then-LLM.

All three go through the same Chat Completions call and the same strict JSON schema, so the
only thing that changes between methods is what the model is shown: extracted text, the page
image, or text that an OCR model read off the page image.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai
from ch08_invoice import Invoice, extract_json, strict_schema
from openai import OpenAI
from PIL import Image
from pydantic import ValidationError
from pypdf import PdfReader

MIN_TEXT_CHARS = 40  # fewer extracted characters than this: the PDF has no real text layer

SYSTEM_RULES = """You extract data from supplier invoices for ShopFlow's accounts-payable team.
Rules:
- Copy what is printed. Do not compute, correct or guess values. SKU and invoice number are
  copied character by character.
- All money is integer cents: $1,234.56 -> 123456, 1.234,56 EUR -> 123456. One unit price per
  line (the per-unit price, not the line amount).
- `date` is the invoice (issue) date, as YYYY-MM-DD. It is NOT the due date, payment date or
  delivery date. Numeric dates follow the supplier's country: a US supplier writes MM/DD/YYYY,
  a UK or German supplier writes DD/MM/YYYY or DD.MM.YYYY.
- `supplier` is the company that issued the invoice, not the customer (ShopFlow Inc.).
- `currency` is the ISO 4217 code (USD, EUR, GBP).
- Include every line item, from every page."""

NAIVE_PROMPT = """Extract this invoice as JSON with the keys: supplier, invoice_no, date \
(YYYY-MM-DD), currency, line_items (a list of objects with sku, desc, qty, unit_price_cents), \
subtotal_cents, tax_cents, total_cents. Return only the JSON."""

OCR_PROMPT = "Text Recognition:"  # the prompt GLM-OCR's model card uses for plain transcription


@dataclass
class Extraction:
    """What one model call produced, plus what it cost."""

    method: str  # text | vision | ocr+llm
    model: str
    invoice: Invoice | None = None
    error: str | None = None
    raw: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str | None = None
    notes: dict = field(default_factory=dict)

    def __add__(self, other: Extraction) -> Extraction:
        """Sum the costs of two chained calls (OCR, then LLM); keep the second's result."""
        return Extraction(
            method=self.method, model=f"{self.model} -> {other.model}", invoice=other.invoice,
            error=other.error, raw=other.raw, latency_s=self.latency_s + other.latency_s,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd, finish_reason=other.finish_reason)


STRICT_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "Invoice", "strict": True, "schema": strict_schema(Invoice)},
}
JSON_OBJECT_FORMAT = {"type": "json_object"}


def pdf_text(path: Path) -> str:
    """Text layer of every page, joined. Empty for image-only PDFs."""
    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


def has_text_layer(text: str) -> bool:
    return len(text.strip()) >= MIN_TEXT_CHARS


def image_data_url(path: Path, max_side: int | None = None) -> str:
    """Base64 data URL for an image file, optionally downscaled so its longest side <= max_side.

    Local vision models have small fixed input sizes and a 4k-token Ollama context, so we send
    them smaller pages. Hosted models get the original pixels.
    """
    if max_side is not None:
        with Image.open(path) as im:
            if max(im.size) > max_side:
                scale = max_side / max(im.size)
                im = im.resize((round(im.width * scale), round(im.height * scale)),
                               Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}[path.suffix.lower()]
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"



def text_messages(text: str, *, system: str = SYSTEM_RULES) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Invoice text:\n\n{text}"}]


def image_messages(path: Path, *, system: str = SYSTEM_RULES,
                   max_side: int | None = None) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract this invoice."},
                {"type": "image_url", "image_url": {"url": image_data_url(path, max_side)}}]}]


def repair_messages(messages: list[dict], previous_raw: str, problems: str) -> list[dict]:
    """Conversation for a second try: the original request, the bad answer, what was wrong."""
    return [*messages,
            {"role": "assistant", "content": previous_raw or "{}"},
            {"role": "user", "content": (
                "That extraction broke these consistency checks: "
                f"{problems}. Look at the document again and return a corrected JSON object. "
                "Only change a value if the document shows it differently.")}]


def call_model(client: OpenAI, model: str, messages: list[dict], *, method: str,
               response_format: dict | None = STRICT_FORMAT, max_tokens: int = 4000,
               **kwargs) -> Extraction:
    """One Chat Completions call -> `Extraction`. Never raises for model/validation problems."""
    ex = Extraction(method=method, model=model)
    t0 = time.perf_counter()
    try:
        kw = dict(kwargs)
        if response_format is not None:
            kw["response_format"] = response_format
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, **kw)
    except openai.APIError as e:
        ex.latency_s = time.perf_counter() - t0
        ex.error = f"api_error: {type(e).__name__}: {str(e)[:200]}"
        return ex
    ex.latency_s = time.perf_counter() - t0
    choice = resp.choices[0]
    ex.raw = choice.message.content or ""
    ex.finish_reason = choice.finish_reason
    if resp.usage:
        ex.prompt_tokens = resp.usage.prompt_tokens
        ex.completion_tokens = resp.usage.completion_tokens
        cost = (getattr(resp.usage, "model_extra", None) or {}).get("cost")
        ex.cost_usd = float(cost) if cost is not None else 0.0  # Ollama: local, no cost field
    try:
        ex.invoice = Invoice.model_validate(extract_json(ex.raw))
    except (ValueError, ValidationError) as e:
        ex.error = f"parse_error: {type(e).__name__}: {str(e)[:200]}"
    return ex


def extract_from_text(client: OpenAI, model: str, text: str, **kw) -> Extraction:
    return call_model(client, model, text_messages(text), method="text", **kw)


def extract_from_image(client: OpenAI, model: str, path: Path, *, max_side: int | None = None,
                       **kw) -> Extraction:
    msgs = image_messages(path, max_side=max_side)
    return call_model(client, model, msgs, method="vision", **kw)


def ocr_page(ollama: OpenAI, path: Path, model: str = "glm-ocr", max_side: int = 1000) -> Extraction:
    """Transcribe a page image with a dedicated OCR model. `raw` holds the text it read.

    The page is downscaled first: at full 1240x1754 px glm-ocr returned almost nothing
    (see the chapter), at ~1000 px it transcribes the whole page.
    """
    messages = [{"role": "user", "content": [
        {"type": "text", "text": OCR_PROMPT},
        {"type": "image_url", "image_url": {"url": image_data_url(path, max_side)}}]}]
    ex = call_model(ollama, model, messages, method="ocr", response_format=None, max_tokens=3000)
    ex.error = ex.error if ex.error and ex.error.startswith("api_error") else None  # not JSON
    return ex


def extract_ocr_then_llm(ollama: OpenAI, llm: OpenAI, model: str, path: Path, **kw) -> Extraction:
    """Two steps: OCR model reads the page, a text LLM structures what it read."""
    ocr = ocr_page(ollama, path)
    if ocr.error:
        return ocr
    second = call_model(llm, model, text_messages(ocr.raw), method="ocr+llm", **kw)
    out = ocr + second
    out.method = "ocr+llm"
    out.notes["ocr_chars"] = len(ocr.raw)
    return out
