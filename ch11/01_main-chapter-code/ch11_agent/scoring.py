"""Did the report rank the injected root cause first? Deterministic keyword check.

We score the hypothesis *statement* only (not its evidence), because evidence for a wrong
hypothesis often mentions the right metric ("the pool is fine").
"""

from __future__ import annotations

from .report import IncidentReport, ranked

# scenario -> list of groups; a hypothesis matches when every group has at least one term in it
RULES: dict[str, list[list[str]]] = {
    "db-pool-exhaustion": [["pool"]],
    "bad-deploy": [["deploy", "release", "rollout", "version"]],
    "payment-provider-timeout": [["timeout", "timed out", "slow", "latency", "degrad"]],
}


def matches(text: str, scenario: str, provider: str = "paypal") -> bool:
    t = text.lower()
    groups = list(RULES[scenario])
    if scenario == "payment-provider-timeout":
        groups.append([provider])
    return all(any(term in t for term in g) for g in groups)


def rank_of_root_cause(report: IncidentReport, scenario: str, provider: str = "paypal") -> int | None:
    """1-based rank of the first matching hypothesis, or None if no hypothesis matches."""
    for i, h in enumerate(ranked(report), 1):
        if matches(h.hypothesis, scenario, provider):
            return i
    return None
