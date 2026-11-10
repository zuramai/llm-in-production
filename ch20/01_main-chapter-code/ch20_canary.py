"""Prompt-version canary: set the percentage, then compare canary against stable from the logs.

    uv run python ch20/01_main-chapter-code/ch20_canary.py set 10        # 10% of sessions on v2
    uv run python ch20/01_main-chapter-code/ch20_canary.py analyze --since 15m
    uv run python ch20/01_main-chapter-code/ch20_canary.py set 0         # rollback

`set` writes a kustomize overlay that merges CANARY_PERCENT into the generated chat ConfigMap
and applies it. The ConfigMap name carries a content hash, so the change rolls the Deployment
and `kubectl rollout undo deployment/chat` restores the previous percentage.

`analyze` reads the one-JSON-line-per-request logs of every chat pod and compares versions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "k8s" / "overlays" / ".generated"


def overlay_yaml(percent: float, source: str, registry: str | None = None,
                 tag: str | None = None) -> str:
    """Kustomization that layers CANARY_PERCENT (and optionally image refs) on k8s/<source>.

    Every `set` must carry the image tag too: an overlay without it would quietly roll the
    Deployment back to the images named in the base.
    """
    if not 0 <= percent <= 100:
        raise ValueError(f"percent must be in [0, 100], got {percent}")
    text = (
        "apiVersion: kustomize.config.k8s.io/v1beta1\n"
        "kind: Kustomization\n"
        "namespace: shopflow\n"
        f"resources: [../../../{source}]\n"
        "configMapGenerator:\n"
        "  - name: chat-config\n"
        "    behavior: merge\n"
        f"    literals: [CANARY_PERCENT={percent:g}]\n"
    )
    if registry and tag:
        text += "images:\n" + "".join(
            f"  - {{name: shopflow-{svc}, newName: {registry}/shopflow-{svc}, newTag: {tag!r}}}\n"
            for svc in ("chat", "sqlbot"))
    return text


def write_overlay(percent: float, source: str = "base", registry: str | None = None,
                  tag: str | None = None) -> Path:
    target = GENERATED / f"canary-{source.replace('/', '-')}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "kustomization.yaml").write_text(overlay_yaml(percent, source, registry, tag))
    return target


def parse_logs(text: str) -> list[dict]:
    """Chat request events from `kubectl logs` output (other lines are ignored)."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") == "chat":
            events.append(ev)
    return events


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def version_stats(events: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for version in sorted({e["prompt_version"] for e in events}):
        evs = [e for e in events if e["prompt_version"] == version]
        ok = [e for e in evs if e["status"] == "ok"]
        out[version] = {
            "requests": len(evs),
            "sessions": len({e["session_id"] for e in evs}),
            "error_rate": round(1 - len(ok) / len(evs), 4),
            "ttft_p95_ms": _p95([e["ttft_ms"] for e in ok if e.get("ttft_ms") is not None]),
            "total_p95_ms": _p95([e["total_ms"] for e in ok]),
            "avg_completion_tokens": round(statistics.mean(
                [e["completion_tokens"] or 0 for e in ok]), 1) if ok else None,
            "cost_per_1k_usd": round(1000 * statistics.mean(
                [e["cost_usd"] or 0 for e in ok]), 4) if ok else None,
        }
    return out


def verdict(stats: dict[str, dict], stable: str, canary: str, *, min_requests: int = 20,
            max_error_delta: float = 0.01, max_latency_ratio: float = 1.25) -> tuple[str, list]:
    """'promote', 'rollback' or 'wait' (not enough canary traffic yet), with reasons."""
    s, c = stats.get(stable), stats.get(canary)
    if not c or c["requests"] < min_requests or not s:
        return "wait", [f"canary has {c['requests'] if c else 0} requests, need {min_requests}"]
    reasons = []
    if c["error_rate"] > s["error_rate"] + max_error_delta:
        reasons.append(f"error rate {c['error_rate']:.2%} vs {s['error_rate']:.2%}")
    if c["total_p95_ms"] and s["total_p95_ms"] and (
            c["total_p95_ms"] > max_latency_ratio * s["total_p95_ms"]):
        reasons.append(f"p95 {c['total_p95_ms']:.0f} ms vs {s['total_p95_ms']:.0f} ms")
    return ("rollback", reasons) if reasons else ("promote", ["within thresholds"])


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(HERE))
    from ch20_ops import kubectl, kubectl_cluster

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set")
    s.add_argument("percent", type=float)
    s.add_argument("--source", default="base", help="k8s/<source> to layer on (base or "
                                                    "overlays/fakellm)")
    s.add_argument("--registry", help="image registry, e.g. ghcr.io/acme")
    s.add_argument("--tag", help="image tag that passed the eval gate")
    a = sub.add_parser("analyze")
    a.add_argument("--since", default="15m")
    a.add_argument("--stable", default="v1")
    a.add_argument("--canary", default="v2")
    args = p.parse_args(argv)

    if args.cmd == "set":
        overlay = write_overlay(args.percent, args.source, args.registry, args.tag)
        print(kubectl_cluster(f"apply -k {overlay}"))
        print(kubectl("rollout status deployment/chat --timeout=180s", timeout=210))
        return 0
    logs = kubectl(f"logs -l app=chat --since={args.since} --tail=-1 --prefix=false "
                   "--max-log-requests=10")
    stats = version_stats(parse_logs(logs))
    decision, reasons = verdict(stats, args.stable, args.canary)
    print(json.dumps({"stats": stats, "decision": decision, "reasons": reasons}, indent=2))
    return {"promote": 0, "wait": 2}.get(decision, 1)


if __name__ == "__main__":
    sys.exit(main())
