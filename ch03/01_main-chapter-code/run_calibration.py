"""Run every judge configuration over the 40 labelled answers and save calibration/judge_*.json.

uv run python ch03/01_main-chapter-code/run_calibration.py [name ...]

Hosted judges go through OpenRouter (`.env`); local judges go to Ollama on the GPU and cost $0.
"""

import sys
import time

from ch03_calibrate import load_calibration, run_judge, save_run
from ch03_faqbot import FAQ_TEXT, ollama_client

from llmprod import chat_client

NAIVE_RUBRIC = "Rate the quality of the assistant's answer from 1 (terrible) to 5 (excellent)."

# name -> (model, local?, rubric or None for the full rubric, reference FAQ, extra request options)
CONFIGS = {
    "luna_naive": ("openai/gpt-6-luna", False, NAIVE_RUBRIC, "(none)", {}),
    "luna_run1": ("openai/gpt-6-luna", False, None, FAQ_TEXT, {}),
    "luna_run2": ("openai/gpt-6-luna", False, None, FAQ_TEXT, {}),
    "luna_run3": ("openai/gpt-6-luna", False, None, FAQ_TEXT, {}),
    "haiku": ("anthropic/claude-haiku-5.5", False, None, FAQ_TEXT, {}),
    "sonnet": ("anthropic/claude-sonnet-5.5", False, None, FAQ_TEXT, {}),
    "qwen_flash": (
        "qwen/qwen3.8-flash",
        False,
        None,
        FAQ_TEXT,
        {"max_tokens": 4000, "reasoning_effort": "low"},
    ),
    # Local judges on the RTX 3090. Ollama's default 4,096-token context holds the ~1.5k-token
    # judge prompt plus the reasoning; `reasoning_effort` keeps gpt-oss from thinking too long.
    "gptoss20b": (
        "gpt-oss:20b",
        True,
        None,
        FAQ_TEXT,
        {"max_tokens": 2000, "reasoning_effort": "low"},
    ),
    "qwen3_8b": ("qwen3:8b", True, None, FAQ_TEXT, {"reasoning_effort": "none"}),
}

if __name__ == "__main__":
    answers, _ = load_calibration()
    for name in sys.argv[1:] or CONFIGS:
        model, local, rubric, faq, opts = CONFIGS[name]
        client = ollama_client() if local else chat_client(timeout=90)
        kwargs = {"rubric": rubric} if rubric else {}
        t0 = time.perf_counter()
        runs = run_judge(client, model, answers, faq, workers=4 if local else 8, **kwargs, **opts)
        wall = time.perf_counter() - t0
        save_run(
            name,
            runs,
            {"model": model, "wall_s": round(wall, 1), "host": "local" if local else "openrouter"},
        )
        print(name, model, f"cost=${sum(j.cost_usd for j in runs.values()):.4f} wall={wall:.1f}s")
