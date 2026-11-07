"""Run every judge configuration over the 40 labelled answers and save calibration/judge_*.json.

uv run python ch03/01_main-chapter-code/run_calibration.py [name ...]
"""

import sys

from ch03_calibrate import load_calibration, run_judge, save_run
from ch03_faqbot import FAQ_TEXT

from llmprod import chat_client

NAIVE_RUBRIC = "Rate the quality of the assistant's answer from 1 (terrible) to 5 (excellent)."

# name -> (model, rubric or None for the full rubric, reference FAQ, extra request options)
CONFIGS = {
    "luna_naive": ("openai/gpt-6-luna", NAIVE_RUBRIC, "(none)", {}),
    "luna_run1": ("openai/gpt-6-luna", None, FAQ_TEXT, {}),
    "luna_run2": ("openai/gpt-6-luna", None, FAQ_TEXT, {}),
    "luna_run3": ("openai/gpt-6-luna", None, FAQ_TEXT, {}),
    "haiku": ("anthropic/claude-haiku-5.5", None, FAQ_TEXT, {}),
    "sonnet": ("anthropic/claude-sonnet-5.5", None, FAQ_TEXT, {}),
    "qwen_flash": (
        "qwen/qwen3.8-flash",
        None,
        FAQ_TEXT,
        {"max_tokens": 4000, "reasoning_effort": "low"},
    ),
}

if __name__ == "__main__":
    answers, _ = load_calibration()
    client = chat_client(timeout=90)
    for name in sys.argv[1:] or CONFIGS:
        model, rubric, faq, opts = CONFIGS[name]
        kwargs = {"rubric": rubric} if rubric else {}
        runs = run_judge(client, model, answers, faq, **kwargs, **opts)
        save_run(name, runs, {"model": model})
        print(name, model, f"cost=${sum(j.cost_usd for j in runs.values()):.4f}")
