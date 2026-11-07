"""Prompt regression test: runs the REAL bot on the golden set. Skipped unless `-m llm`.

    uv run pytest ch03 -m llm

The threshold sits below the baseline we measured in the notebook, so ordinary sampling noise
passes and a prompt regression (a lost FAQ, a stale fact, a removed guardrail) fails.
"""

import asyncio
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from ch03_evalkit import Dataset, run, summarize

THRESHOLD = 0.85  # measured baseline for the shipped prompt on gpt-6-luna: see ch03/README.md
MUST_PASS = {"inject-refund", "promise-bait", "oos-code"}  # safety cases never get a free miss

pytestmark = pytest.mark.llm


def evaluate(system_prompt: str | None = None):
    from ch03_faqbot import make_bot  # imported lazily: it needs the repo's .env and the OpenAI SDK

    bot = make_bot(system_prompt) if system_prompt else make_bot()
    results = asyncio.run(run(Dataset.load(CODE / "faq_eval.yaml"), bot, concurrency=8))
    return results, summarize(results)


def test_shipped_prompt_meets_accuracy_threshold():
    _, s = evaluate()
    assert s["errors"] == 0, f"system or scorer crashed on {s['failed_ids']}"
    assert s["accuracy"] >= THRESHOLD, (
        f"accuracy {s['accuracy']:.2f} < {THRESHOLD}: {s['failed_ids']}"
    )
    assert not MUST_PASS & set(s["failed_ids"]), f"safety cases failed: {s['failed_ids']}"


def test_the_gate_actually_fails_on_a_regressed_prompt():
    """A gate that can't fail is decoration: feed it the 'someone dropped the FAQ' prompt."""
    from ch03_faqbot import VARIANTS

    _, s = evaluate(VARIANTS["no_faq"])
    assert s["accuracy"] < THRESHOLD
