"""Matplotlib charts for Chapter 22, drawn from results/run_all.json."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGES = Path(__file__).resolve().parents[1] / "images"
AMBER, BLUE, GREY, RED = "#e0a030", "#2f6fde", "#9aa3b2", "#c8453c"


def _save(fig, name: str) -> Path:
    IMAGES.mkdir(exist_ok=True)
    path = IMAGES / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def agent_runaway(d3: dict) -> Path:
    """Prompt tokens per model call: unguarded luna vs Haiku vs luna with the guard."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for key, label, color in (("luna", "gpt-6-luna, no guard", RED),
                              ("haiku", "claude-haiku-5.5, no guard", GREY),
                              ("guarded", "gpt-6-luna + Guard", BLUE)):
        tokens = [t["prompt_tokens"] for t in d3[key]["trace"]]
        ax.plot(range(1, len(tokens) + 1), tokens, marker="o", ms=3.5, label=label, color=color)
    ax.set_xlabel("model call in one conversation")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("prompt tokens sent")
    ax.set_title("D3: a search that always has a next page")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    return _save(fig, "22-agent-runaway.png")


def retry_storm(f1: dict) -> Path:
    """Arrivals at the fake provider per 100 ms: SDK retries vs a client-side bucket."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.2), sharex=True)
    for ax, (key, color) in zip(axes, (("sdk", RED), ("client bucket", BLUE)), strict=True):
        bins = Counter(int(t * 10) for t in f1[key]["arrivals"])
        xs = range(max(bins) + 1)
        ax.bar([x / 10 for x in xs], [bins.get(x, 0) for x in xs], width=0.09, color=color)
        ax.axhline(2, color=GREY, lw=1, ls="--")
        ax.set_ylabel("calls / 100 ms")
        ax.set_title(f"{key}: {f1[key]['upstream_calls']} calls, "
                     f"{f1[key]['rejected_429']} rejected", fontsize=10, loc="left")
        ax.grid(alpha=0.3, axis="y")
    axes[1].set_xlabel("seconds since the burst (dashed: provider limit, 20/s)")
    fig.suptitle("F1: 100 requests at once against a 20 req/s limit", fontsize=11)
    fig.tight_layout()
    return _save(fig, "22-retry-storm.png")


def prompt_cache(i1: dict) -> Path:
    """Cost of calls 2 and 3 with and without a stable prefix."""
    summary = i1["summary"]
    labels = [k.replace(", ", "\n", 1) for k in summary]
    costs = [v["cost_calls_2_3"] * 1000 for v in summary.values()]
    colors = [RED, BLUE, RED, BLUE]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.barh(labels, costs, color=colors)
    for y, c in enumerate(costs):
        ax.text(c, y, f"  ${c:.3f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"cost of calls 2 and 3, in \$ per 1,000 conversations")
    ax.set_title("I1: same ~4.6k–6.9k token policy prefix, three questions")
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, "22-prompt-cache.png")


def confidence(e2: dict) -> Path:
    """Verbalized confidence vs 3-sample agreement, for right and wrong answers."""
    rows = e2["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)
    for ax, key, title in ((axes[0], "confidence", "self-reported confidence"),
                           (axes[1], "agreement", "agreement of 3 samples")):
        for ok, color, label in ((True, BLUE, "right"), (False, RED, "wrong")):
            vals = [r[key] for r in rows if r["ok"] == ok]
            jitter = [((i * 37) % 11 - 5) / 40 for i in range(len(vals))]
            ax.scatter(vals, [(1 if ok else 0) + j for j in jitter], color=color, alpha=0.6,
                       s=40, label=f"{label} ({len(vals)})")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_yticks([0, 1], ["wrong", "right"])
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("E2: which signal separates luna's right answers from wrong ones?", fontsize=11)
    fig.tight_layout()
    return _save(fig, "22-confidence.png")


def overview(results: dict) -> Path:
    """Per layer: entries, failures reproduced, fixes holding."""
    layers: dict[str, list[int]] = {}
    for r in results.values():
        row = layers.setdefault(r["layer"], [0, 0, 0])
        row[0] += 1
        row[1] += bool(r["reproduced"])
        row[2] += bool(r["fixed"])
    names = list(layers)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    xs = range(len(names))
    for off, idx, color, label in ((-0.27, 0, GREY, "entries"), (0, 1, AMBER, "failure reproduced"),
                                   (0.27, 2, BLUE, "fix held")):
        ax.bar([x + off for x in xs], [layers[n][idx] for n in names], width=0.26, color=color,
               label=label)
    ax.set_xticks(list(xs), [n.replace(" & ", "\n& ").replace(" ", "\n") for n in names],
                  fontsize=8.5)
    ax.set_ylabel("entries")
    ax.set_title("All repros, one run on this host")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "22-overview.png")
