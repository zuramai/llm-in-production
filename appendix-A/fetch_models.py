"""Snapshot OpenRouter's public model catalog and summarize it as Markdown.

    uv run python appendix-A/fetch_models.py            # writes appendix-A/models-<date>.csv

No API key needed. Re-run it whenever you read this appendix: model lists and prices change weekly.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/models"
HERE = Path(__file__).resolve().parent


def per_million(price: str | None) -> float | None:
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    return None if value < 0 else value * 1_000_000


def load() -> list[dict]:
    rows = []
    for m in httpx.get(URL, timeout=30).json()["data"]:
        params = set(m.get("supported_parameters") or [])
        arch = m.get("architecture") or {}
        rows.append({
            "id": m["id"],
            "vendor": m["id"].split("/")[0],
            "created": datetime.fromtimestamp(m.get("created", 0), UTC).date().isoformat(),
            "context_length": m.get("context_length") or 0,
            "usd_in_per_m": per_million(m["pricing"].get("prompt")),
            "usd_out_per_m": per_million(m["pricing"].get("completion")),
            "usd_cache_read_per_m": per_million(m["pricing"].get("input_cache_read")),
            "input_modalities": "+".join(arch.get("input_modalities") or []),
            "tools": "tools" in params,
            "structured_outputs": "structured_outputs" in params,
            "temperature": "temperature" in params,
            "reasoning_mandatory": bool((m.get("reasoning") or {}).get("mandatory")),
            "expiration_date": m.get("expiration_date") or "",
        })
    return rows


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def money(x: float | None) -> str:
    return "n/a" if x is None else f"${x:,.2f}"


def main() -> None:
    rows = load()
    today = date.today().isoformat()
    out = HERE / f"models-{today}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    paid = [r for r in rows if r["usd_in_per_m"] and r["usd_out_per_m"]]
    free = [r for r in rows if r["usd_in_per_m"] == 0 and r["usd_out_per_m"] == 0]
    print(f"## Snapshot {today}: {len(rows)} models from {len({r['vendor'] for r in rows})} vendors"
          f" ({len(free)} free, {len(paid)} paid) → {out.name}\n")

    print("### Models per vendor (top 12)\n")
    print(table(["vendor", "models"], Counter(r["vendor"] for r in rows).most_common(12)))

    print("\n### Price distribution of paid models ($ per 1M tokens)\n")
    qs = [10, 25, 50, 75, 90]
    pin = statistics.quantiles([r["usd_in_per_m"] for r in paid], n=100)
    pout = statistics.quantiles([r["usd_out_per_m"] for r in paid], n=100)
    print(table(["percentile"] + [f"p{q}" for q in qs],
                [["input"] + [money(pin[q - 1]) for q in qs],
                 ["output"] + [money(pout[q - 1]) for q in qs]]))
    ratio = statistics.median(r["usd_out_per_m"] / r["usd_in_per_m"] for r in paid)
    print(f"\nMedian output/input price ratio: {ratio:.1f}×")

    print("\n### Capabilities (share of all models)\n")
    n = len(rows)
    caps = [
        ("tool calling", sum(r["tools"] for r in rows)),
        ("structured outputs (json_schema)", sum(r["structured_outputs"] for r in rows)),
        ("accepts temperature", sum(r["temperature"] for r in rows)),
        ("reasoning is mandatory", sum(r["reasoning_mandatory"] for r in rows)),
        ("image input", sum("image" in r["input_modalities"] for r in rows)),
        ("context ≥ 1M tokens", sum(r["context_length"] >= 1_000_000 for r in rows)),
    ]
    print(table(["capability", "models", "share"], [[c, k, f"{k / n:.0%}"] for c, k in caps]))

    print("\n### Cheapest paid models with tools + structured outputs + ≥128k context\n")
    prod = sorted((r for r in paid if r["tools"] and r["structured_outputs"]
                   and r["context_length"] >= 128_000),
                  key=lambda r: (r["usd_in_per_m"] + 3 * r["usd_out_per_m"]))[:12]
    print(table(["model", "context", "$ in", "$ out", "$ cache read"],
                [[r["id"], f"{r['context_length']:,}", money(r["usd_in_per_m"]),
                  money(r["usd_out_per_m"]), money(r["usd_cache_read_per_m"])] for r in prod]))

    print("\n### Models used in this book\n")
    book = ["openai/gpt-6-luna", "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5",
            "qwen/qwen3.8-flash"]
    by_id = {r["id"]: r for r in rows}
    print(table(["model", "context", "$ in", "$ out", "temperature", "reasoning mandatory"],
                [[m, f"{by_id[m]['context_length']:,}", money(by_id[m]["usd_in_per_m"]),
                  money(by_id[m]["usd_out_per_m"]), by_id[m]["temperature"],
                  by_id[m]["reasoning_mandatory"]] for m in book if m in by_id]))

    expiring = sorted((r for r in rows if r["expiration_date"]), key=lambda r: r["expiration_date"])
    print(f"\n### Scheduled expirations ({len(expiring)} models)\n")
    print(table(["model", "expires"], [[r["id"], r["expiration_date"]] for r in expiring[:20]]))

    newest = sorted(rows, key=lambda r: r["created"], reverse=True)[:10]
    print("\n### Ten newest listings\n")
    print(table(["model", "listed", "$ in", "$ out"],
                [[r["id"], r["created"], money(r["usd_in_per_m"]), money(r["usd_out_per_m"])]
                 for r in newest]))


if __name__ == "__main__":
    main()
