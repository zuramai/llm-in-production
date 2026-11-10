"""Investigate an alert from the command line and print the draft postmortem.

    uv run python investigate.py "5xx ratio on checkout above 5% for 5 minutes"
    uv run python investigate.py "..." --model gpt-oss:20b --out reports/incident.md

In production the same call sits behind an Alertmanager webhook and posts the Markdown to the
incident channel. It never acts on its own proposals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ch11_agent.agent import run_agent
from ch11_agent.report import render_markdown

from llmprod import get_settings


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("alert", help="the alert text, as the pager would deliver it")
    p.add_argument("--model", help="'vendor/model' via LLM_BASE_URL, or 'name:tag' via Ollama")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--out", type=Path, help="also write the Markdown to this file")
    a = p.parse_args()
    model = a.model or get_settings().llm_model
    res = run_agent(a.alert, model, max_steps=a.max_steps)
    if res.report is None:
        print(f"investigation failed: {res.error}", file=sys.stderr)
        return 1
    md = render_markdown(res.report, alert=a.alert, model=model, steps=res.steps,
                         cost_usd=res.cost_usd)
    if a.out:
        a.out.write_text(md)
    print(md)
    print(f"[{len(res.trace)} tool calls, {res.prompt_tokens} prompt tokens, "
          f"{res.latency_s:.1f} s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
