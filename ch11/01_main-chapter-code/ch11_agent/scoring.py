"""Did the report rank the injected root cause first? Deterministic keyword check.

We score the hypothesis *statement* only (not its evidence), because evidence for a wrong
hypothesis often mentions the right metric ("the pool is fine").
"""

from __future__ import annotations

import re

from .report import IncidentReport, ranked

# scenario -> list of groups; a hypothesis matches when every group has at least one term in it
RULES: dict[str, list[list[str]]] = {
    "db-pool-exhaustion": [["pool"]],
    # the injected release (inject_fault.py: "Apply promo discounts in checkout pricing") or its
    # symptom (TypeError on None) names the culprit even without the word "deploy"
    "bad-deploy": [["deploy", "release", "rollout", "version", "regression",
                    "promo", "discount", "nonetype", "null"]],
    "payment-provider-timeout": [["timeout", "timed out", "slow", "latency", "degrad"]],
}
# blaming a release when none was injected, or the healthy gift-message release (inject_fault.py)
# by feature name: the human would roll back a healthy deploy, so it isn't a hit
BLAMES_RELEASE = ["deploy", "release", "rollout", "regression"]
HEALTHY_RELEASE = ["gift-message", "gift message", "gift\u2011message"]
VERSION = re.compile(r"v\d{4}\.\d{2}\.\d{2}-\d+")


def matches(text: str, scenario: str, provider: str = "paypal",
            bad_version: str | None = None) -> bool:
    t = text.lower()
    named = set(VERSION.findall(t))
    culprit = bad_version.lower() if scenario == "bad-deploy" and bad_version else None
    if culprit in named:
        return True  # "A bug in v2026.11.07-2's promo pricing" names the culprit exactly
    if named - {culprit}:
        # blames a release that didn't cause it ("pool exhausted *because of* v…-1"): the human
        # would roll back a healthy deploy, so it doesn't count even if the mechanism is right
        return False
    if any(term in t for term in HEALTHY_RELEASE):
        return False
    if scenario != "bad-deploy" and any(term in t for term in BLAMES_RELEASE):
        return False
    groups = list(RULES[scenario])
    if scenario == "payment-provider-timeout":
        groups.append([provider])
    return all(any(term in t for term in g) for g in groups)


def first_match(hypotheses: list[str], scenario: str, provider: str = "paypal",
                bad_version: str | None = None) -> int | None:
    """1-based position of the first matching statement in an already-ranked list."""
    for i, h in enumerate(hypotheses, 1):
        if matches(h, scenario, provider, bad_version):
            return i
    return None


def rank_of_root_cause(report: IncidentReport, scenario: str, provider: str = "paypal",
                       bad_version: str | None = None) -> int | None:
    """1-based rank of the first matching hypothesis, or None if no hypothesis matches."""
    return first_match([h.hypothesis for h in ranked(report)], scenario, provider, bad_version)
