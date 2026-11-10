"""A5: unicode and tokenization surprises: curly quotes, NFD, whitespace, tokenizer gaps."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import tiktoken
from ch22_common import HAIKU, LUNA, Result, chat, timed

ID, TITLE, LAYER, KIND = "A5", "Unicode and tokenization surprises", "API & tokens", "hosted"

REPO = Path(__file__).resolve().parents[3]
QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                        "\u00a0": " ", "\u2013": "-", "\u2014": "-"})

# The same customer message in four languages (translations by the author).
SAMPLES = {
    "English": "My order hasn't arrived yet. Can you check where it is and refund the shipping?",
    "Indonesian": "Pesanan saya belum sampai. Bisakah Anda cek posisinya dan kembalikan ongkos kirim?",
    "Japanese": "注文した商品がまだ届いていません。どこにあるか確認して、送料を返金してもらえますか？",
    "Emoji-heavy": "My order 📦 hasn't arrived 😡😡 Can you check where it is 🔎 and refund shipping 💸?",
}


def normalize_text(text: str) -> str:
    """Canonical form for comparisons and cache keys (not for what you send to the model)."""
    text = unicodedata.normalize("NFKC", text).translate(QUOTES)
    return re.sub(r"\s+", " ", text).strip().casefold()


def cache_key(prompt: str) -> str:
    return hashlib.sha256(normalize_text(prompt).encode()).hexdigest()[:16]


@timed
def run() -> Result:
    enc = tiktoken.get_encoding("o200k_base")
    per_lang = {k: {"chars": len(v), "tokens": len(enc.encode(v))} for k, v in SAMPLES.items()}

    reply = "Sorry, that order wasn\u2019t found."  # what the model actually wrote (curly)
    nfd = unicodedata.normalize("NFD", "Café Dürüm")
    checks = {
        "curly apostrophe: \"wasn't found\" in reply": "wasn't found" in reply,
        "NFD vs NFC: 'Café Dürüm' == nfd": "Café Dürüm" == nfd,
        "extra space: same sha256": hashlib.sha256(b"refund policy?").hexdigest()
        == hashlib.sha256(b"refund  policy?").hexdigest(),
    }
    fixed_checks = {
        "curly": normalize_text("wasn't found") in normalize_text(reply),
        "nfd": normalize_text("Café Dürüm") == normalize_text(nfd),
        "space": cache_key("refund policy?") == cache_key("refund  policy?"),
    }

    doc = (REPO / "datasets/shopflow/docs/policies/refund-policy.md").read_text()
    billed = {}
    for model in (LUNA, HAIKU):
        r = chat(model, [{"role": "user", "content": doc + "\n\nReply OK."}], max_tokens=16,
                 reasoning_effort="none")
        billed[model] = r.usage.prompt_tokens
    ratio = billed[HAIKU] / billed[LUNA]
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=not any(checks.values()) and ratio > 1.1,
        fixed=all(fixed_checks.values()),
        evidence=(
            f"naive equality all False ({len(checks)} checks); same refund-policy.md billed "
            f"{billed[LUNA]} prompt tokens on luna vs {billed[HAIKU]} on haiku ({ratio:.2f}x); "
            f"o200k tokens/char: " + ", ".join(
                f"{k} {v['tokens'] / v['chars']:.2f}" for k, v in per_lang.items())
        ),
        data={"per_lang": per_lang, "naive_checks": checks, "billed_prompt_tokens": billed},
    )
