"""Inject (or clear) a fault in the simulated checkout service.

    uv run python inject_fault.py --scenario db-pool-exhaustion
    uv run python inject_fault.py --scenario bad-deploy        # also appends to deploys.json
    uv run python inject_fault.py --scenario payment-provider-timeout [--provider paypal]
    uv run python inject_fault.py --scenario none               # reset: healthy + baseline deploys

`none` rewrites deploys.json with a baseline history relative to now, so every run starts clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

DEPLOYS = Path(__file__).with_name("deploys.json")
SCENARIOS = ["none", "db-pool-exhaustion", "bad-deploy", "payment-provider-timeout"]


def _record(version: str, when: datetime, author: str, summary: str) -> dict:
    return {"version": version, "sha": hashlib.sha1(version.encode()).hexdigest()[:7],
            "deployed_at": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "author": author,
            "summary": summary}


def baseline(now: datetime) -> list[dict]:
    """Two ordinary deploys. The newer one is a harmless distractor (a few hours old)."""
    older, newer = now - timedelta(hours=30), now - timedelta(hours=5)
    return [
        _record(f"v{older:%Y.%m.%d}-2", older, "priya", "Round cart totals with Decimal"),
        _record(f"v{newer:%Y.%m.%d}-1", newer, "marcus", "Add gift-message field to checkout"),
    ]


def next_version(history: list[dict], now: datetime) -> str:
    n = sum(1 for d in history if d["version"].startswith(f"v{now:%Y.%m.%d}-")) + 1
    return f"v{now:%Y.%m.%d}-{n}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=SCENARIOS)
    p.add_argument("--provider", default="paypal", choices=["stripe", "paypal"])
    p.add_argument("--base-url", default="http://localhost:8011")
    a = p.parse_args()
    now = datetime.now(UTC)

    if a.scenario == "none":
        history = baseline(now)
        version = history[-1]["version"]
    else:
        history = json.loads(DEPLOYS.read_text()) if DEPLOYS.exists() else baseline(now)
        version = history[-1]["version"]
    if a.scenario == "bad-deploy":
        version = next_version(history, now)
        history.append(_record(version, now, "dana", "Apply promo discounts in checkout pricing"))
    DEPLOYS.write_text(json.dumps(history, indent=2) + "\n")

    r = httpx.post(f"{a.base_url}/admin/state", timeout=5, json={
        "scenario": a.scenario, "deploy_version": version, "provider": a.provider})
    r.raise_for_status()
    print(json.dumps({"injected_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), **r.json()}))


if __name__ == "__main__":
    main()
