"""J1: models get retired on a schedule, and aliases move under you without a code change."""

from __future__ import annotations

import datetime as dt

import openai
from ch22_common import HAIKU, LUNA, Result, chat, model_catalog, timed

ID, TITLE, LAYER, KIND = "J1", "Model deprecation and silent drift", "Ops", "hosted"

RETIRED = ["anthropic/claude-3-opus", "openai/gpt-4.5-preview"]  # both were flagship once

# What we deployed against, recorded at release time (Chapter 20's eval gate ran on these).
PINNED = {LUNA: "openai/gpt-6-luna-20260922", HAIKU: "anthropic/claude-haiku-5.5-20261007"}


def model_health(catalog: dict, pinned: dict[str, str], today: dt.date,
                 horizon_days: int = 60) -> list[str]:
    """Problems to page on: model gone, retiring soon, or the alias now points elsewhere."""
    problems = []
    for model, snapshot in pinned.items():
        meta = catalog.get(model)
        if meta is None:
            problems.append(f"{model}: not in the catalog any more")
            continue
        if meta.get("expiration_date"):
            left = (dt.date.fromisoformat(meta["expiration_date"]) - today).days
            if left <= horizon_days:
                problems.append(f"{model}: retires {meta['expiration_date']} ({left} days)")
        if meta.get("canonical_slug") != snapshot:
            problems.append(f"{model}: now serves {meta.get('canonical_slug')}, we pinned "
                            f"{snapshot}: re-run the eval gate")
    return problems


@timed
def run() -> Result:
    cat = model_catalog()
    today = dt.date(2026, 11, 8)
    expiring = sorted((m["expiration_date"], mid) for mid, m in cat.items()
                      if m.get("expiration_date"))
    soon = [x for x in expiring
            if (dt.date.fromisoformat(x[0]) - today).days <= 60]
    errors = {}
    for mid in RETIRED:
        try:
            chat(mid, [{"role": "user", "content": "hi"}], max_tokens=5)
            errors[mid] = "200"
        except openai.APIStatusError as e:
            errors[mid] = f"{e.status_code}"
    r = chat(LUNA, [{"role": "user", "content": "hi"}], max_tokens=16, reasoning_effort="none")
    # Simulate next month's catalog: the provider moved the alias to a new snapshot.
    moved = {k: dict(v) for k, v in cat.items() if k in PINNED}
    moved[LUNA]["canonical_slug"] = "openai/gpt-6-luna-20261201"
    now_ok = model_health(cat, PINNED, today)
    after_move = model_health(moved, PINNED, today)
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=bool(soon) and all(v.startswith("404") for v in errors.values())
        and r.model == LUNA,
        fixed=not now_ok and any("re-run the eval gate" in p for p in after_move),
        evidence=f"OpenRouter catalog: {len(cat)} models, {len(expiring)} with an expiration date, "
                 f"{len(soon)} within 60 days (first: {soon[0][1]} on {soon[0][0]}); retired ids "
                 f"-> {errors}; response.model for luna = {r.model!r} (no snapshot), catalog "
                 f"slug {cat[LUNA]['canonical_slug']!r}; model_health today: "
                 f"{now_ok or 'ok'}; after a simulated alias move: {after_move}",
        data={"n_models": len(cat), "expiring": expiring, "errors": errors},
    )
