"""The four tools of the incident agent, plus the output summarizers that keep them small.

Every tool returns a *string* (what the model sees). Errors come back as strings that start with
"ERROR:", so the model can fix a bad query instead of the loop crashing.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

CODE_DIR = Path(__file__).resolve().parents[1]
DOCS = CODE_DIR.parents[1] / "datasets" / "shopflow" / "docs"
PROM_URL = "http://localhost:9091"
LOKI_URL = "http://localhost:3101"
DEPLOYS_PATH = CODE_DIR / "deploys.json"
MAX_TOOL_CHARS = 2500  # hard cap on what any tool may return to the model


# ---------- helpers ----------
def parse_range(s: str) -> timedelta:
    """'90s', '15m', '2h', '1d' -> timedelta. Raises ValueError with a model-readable message."""
    m = re.fullmatch(r"\s*(\d+)\s*([smhd])\s*", s or "")
    if not m:
        raise ValueError(f"bad range {s!r}: use a number plus s, m, h or d, e.g. '15m'")
    n, unit = int(m.group(1)), m.group(2)
    return timedelta(**{{"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[unit]: n})


def truncate(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit("\n", 1)[0]
    return f"{cut}\n... [truncated: {len(text) - len(cut)} more chars. Narrow the query.]"


def _fmt(v: float) -> str:
    if math.isnan(v):
        return "NaN"
    return f"{v:.4g}"


def _labels(metric: dict) -> str:
    items = {k: v for k, v in metric.items() if k not in ("__name__", "job", "instance")}
    return "{" + ",".join(f"{k}={v}" for k, v in sorted(items.items())) + "}" if items else "{}"


# ---------- summarizers (pure functions: unit-tested without a network) ----------
def summarize_matrix(result: list[dict], max_series: int = 8, points: int = 6) -> str:
    """Prometheus range result -> one line per series: first/min/max/last plus a few samples."""
    if not result:
        return ("no data. The metric may not exist or has no samples in this window. "
                "Check the metric catalog in your instructions.")
    lines = []
    for s in result[:max_series]:
        vals = [(float(t), float(v)) for t, v in s["values"] if v not in ("NaN", "+Inf", "-Inf")]
        if not vals:
            lines.append(f"{_labels(s['metric'])}: only NaN/Inf samples")
            continue
        ys = [v for _, v in vals]
        idx = sorted({round(i * (len(vals) - 1) / max(points - 1, 1)) for i in range(points)})
        sample = " ".join(_fmt(vals[i][1]) for i in idx)
        lines.append(f"{_labels(s['metric'])}: first={_fmt(ys[0])} min={_fmt(min(ys))} "
                     f"max={_fmt(max(ys))} last={_fmt(ys[-1])} | samples over window: {sample}")
    if len(result) > max_series:
        lines.append(f"... {len(result) - max_series} more series omitted; add a sum by (...)")
    return "\n".join(lines)


def summarize_vector(result: list[dict], max_series: int = 8) -> str:
    if not result:
        return "no data."
    lines = [f"{_labels(s['metric'])}: {_fmt(float(s['value'][1]))}" for s in result[:max_series]]
    if len(result) > max_series:
        lines.append(f"... {len(result) - max_series} more series omitted")
    return "\n".join(lines)


def summarize_logs(streams: list[dict], max_groups: int = 6) -> str:
    """Loki streams -> counts per distinct (level, msg, route, status, version, provider, error)."""
    rows = []
    for st in streams:
        for ts, line in st["values"]:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                d = {"msg": line[:120]}
            rows.append((int(ts), st["stream"].get("level", "?"), d))
    if not rows:
        return "no log lines matched in this window."
    groups: dict[tuple, list] = {}
    for ts, level, d in rows:
        key = (level, d.get("msg"), d.get("route"), d.get("status"),
               d.get("deploy_version"), d.get("provider"), d.get("error"))
        groups.setdefault(key, []).append(ts)
    out = [f"{len(rows)} log lines in {len(groups)} distinct patterns (most frequent first):"]
    for key, tss in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:max_groups]:
        names = ("level", "msg", "route", "status", "deploy_version", "provider", "error")
        desc = " ".join(f"{n}={v}" for n, v in zip(names, key, strict=True) if v is not None)
        first = datetime.fromtimestamp(min(tss) / 1e9, UTC).strftime("%H:%M:%SZ")
        last = datetime.fromtimestamp(max(tss) / 1e9, UTC).strftime("%H:%M:%SZ")
        out.append(f"- x{len(tss)} [{first}..{last}] {desc}")
    if len(groups) > max_groups:
        out.append(f"... {len(groups) - max_groups} rarer patterns omitted")
    return "\n".join(out)


def raw_logs(streams: list[dict], limit: int = 200) -> str:
    """The naive alternative: dump raw lines. Used only to show the context-size problem."""
    lines = [f"{ts} {line}" for st in streams for ts, line in st["values"]]
    return "\n".join(lines[:limit])


# ---------- tools ----------
def query_prometheus(promql: str, range: str = "15m", *, summarize: bool = True) -> str:
    """Range query over the last `range` (step chosen automatically)."""
    try:
        span = parse_range(range)
        end = time.time()
        step = max(5, int(span.total_seconds() / 60))
        r = httpx.get(f"{PROM_URL}/api/v1/query_range", timeout=15, params={
            "query": promql, "start": end - span.total_seconds(), "end": end, "step": step})
        body = r.json()
    except (ValueError, httpx.HTTPError) as exc:
        return f"ERROR: {exc}"
    if body.get("status") != "success":
        return f"ERROR: {body.get('errorType')}: {body.get('error')}"
    result = body["data"]["result"]
    if not summarize:
        return json.dumps(result)  # naive: the whole matrix, every sample
    return truncate(summarize_matrix(result))


def query_loki(logql: str, range: str = "15m", limit: int = 200, *, summarize: bool = True) -> str:
    try:
        span = parse_range(range)
        end_ns = time.time_ns()
        r = httpx.get(f"{LOKI_URL}/loki/api/v1/query_range", timeout=15, params={
            "query": logql, "start": end_ns - int(span.total_seconds() * 1e9), "end": end_ns,
            "limit": min(limit, 1000), "direction": "backward"})
    except (ValueError, httpx.HTTPError) as exc:
        return f"ERROR: {exc}"
    if r.status_code != 200:
        return f"ERROR: Loki {r.status_code}: {r.text[:300]}"
    streams = r.json()["data"]["result"]
    if not summarize:
        return raw_logs(streams, limit=limit)
    return truncate(summarize_logs(streams))


def get_deploys(since: str = "24h", *, now: datetime | None = None, path: Path = DEPLOYS_PATH) -> str:
    try:
        span = parse_range(since)
    except ValueError as exc:
        return f"ERROR: {exc}"
    now = now or datetime.now(UTC)
    deploys = json.loads(path.read_text()) if path.exists() else []
    rows = []
    for d in sorted(deploys, key=lambda d: d["deployed_at"], reverse=True):
        at = datetime.fromisoformat(d["deployed_at"])
        if now - at <= span:
            mins = int((now - at).total_seconds() // 60)
            rows.append(f"- {d['version']} sha={d['sha']} deployed_at={d['deployed_at']} "
                        f"({mins} min ago) by {d['author']}: {d['summary']}")
    return "\n".join(rows) if rows else f"no deploys in the last {since}."


_WORD = re.compile(r"[a-z0-9_]+")


def _chunks(path: Path) -> list[tuple[str, str]]:
    """Split a Markdown file into (heading, text) sections."""
    out, head, buf = [], path.stem, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            if buf:
                out.append((head, "\n".join(buf).strip()))
            head, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if buf:
        out.append((head, "\n".join(buf).strip()))
    return [(h, t) for h, t in out if t]


def search_runbooks(query: str, k: int = 3, *, docs: Path = DOCS) -> str:
    """Keyword search over runbooks and the handbook: score sections by term overlap."""
    terms = Counter(_WORD.findall(query.lower()))
    if not terms:
        return "ERROR: empty query"
    scored = []
    for path in sorted([*(docs / "runbooks").glob("*.md"), *(docs / "handbook").glob("*.md")]):
        for head, text in _chunks(path):
            words = _WORD.findall((head + " " + text).lower())
            bag, head_words = Counter(words), set(_WORD.findall(head.lower()))
            score = sum(min(bag[t], 3) * (1 + 2 * (t in head_words)) for t in terms) / (
                len(words) ** 0.3 + 1)
            if score > 0:
                scored.append((score, path.relative_to(docs), head, text))
    if not scored:
        return "no matching runbook sections."
    scored.sort(key=lambda s: -s[0])
    parts = [f"[{p}] ## {h}\n{t[:600]}" for _, p, h, t in scored[:k]]
    return truncate("\n\n".join(parts))


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "query_prometheus",
        "description": "Run a PromQL range query over the last `range` and get a compact summary "
                       "per series (first/min/max/last and a few samples). Use sum by (...) to "
                       "keep results small.",
        "parameters": {"type": "object", "properties": {
            "promql": {"type": "string"},
            "range": {"type": "string", "description": "e.g. '10m', '1h'"}},
            "required": ["promql", "range"]}}},
    {"type": "function", "function": {
        "name": "query_loki",
        "description": "Run a LogQL query over the last `range`. Returns line counts grouped by "
                       "distinct pattern (msg, route, status, deploy_version, provider, error).",
        "parameters": {"type": "object", "properties": {
            "logql": {"type": "string"},
            "range": {"type": "string", "description": "e.g. '10m', '1h'"}},
            "required": ["logql", "range"]}}},
    {"type": "function", "function": {
        "name": "get_deploys",
        "description": "List checkout deploys (version, sha, deployed_at UTC, author, summary) "
                       "within the last `since`.",
        "parameters": {"type": "object", "properties": {
            "since": {"type": "string", "description": "e.g. '24h', '3d'"}},
            "required": ["since"]}}},
    {"type": "function", "function": {
        "name": "search_runbooks",
        "description": "Keyword search over ShopFlow runbooks and the on-call handbook.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]


def dispatch(name: str, args: dict, *, summarize: bool = True) -> str:
    try:
        if name == "query_prometheus":
            return query_prometheus(args["promql"], args.get("range", "15m"), summarize=summarize)
        if name == "query_loki":
            return query_loki(args["logql"], args.get("range", "15m"), summarize=summarize)
        if name == "get_deploys":
            return get_deploys(args.get("since", "24h"))
        if name == "search_runbooks":
            return search_runbooks(args["query"])
    except KeyError as exc:
        return f"ERROR: missing argument {exc}"
    return f"ERROR: unknown tool {name!r}"
