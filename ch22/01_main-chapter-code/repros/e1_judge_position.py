"""E1: a pairwise LLM judge that must pick A or B leans on seating order when answers are equal."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ch22_common import LUNA, Result, chat, timed

ID, TITLE, LAYER, KIND = "E1", "LLM judge position bias", "Evals", "hosted"

# 12 pairs of two samples from the same bot (copied from ch03): no real quality difference.
PAIRS = json.loads((Path(__file__).resolve().parents[1] / "data/judge_pairs_same.json")
                   .read_text())
SCHEMA = {"type": "json_schema", "json_schema": {"name": "pair", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "properties": {"winner": {"type": "string", "enum": ["A", "B"]}}, "required": ["winner"]}}}
SYSTEM = "You compare two customer-support answers. Pick the better one: A or B."


def ask(question: str, a: str, b: str) -> str:
    user = f"Question: {question}\n\n<answer_A>\n{a}\n</answer_A>\n<answer_B>\n{b}\n</answer_B>"
    r = chat(LUNA, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
             max_tokens=40, response_format=SCHEMA, reasoning_effort="none")
    return json.loads(r.choices[0].message.content)["winner"]


def swap_verdict(winner_xy: str, winner_yx: str) -> str:
    """x shown as A first, then y shown as A. Only a win that survives the swap counts."""
    first = {"A": "x", "B": "y"}[winner_xy]
    second = {"A": "y", "B": "x"}[winner_yx]
    return first if first == second else "tie"


@timed
def run() -> Result:
    def both(p: dict) -> tuple[str, str]:
        return ask(p["question"], p["x"], p["y"]), ask(p["question"], p["y"], p["x"])

    with ThreadPoolExecutor(8) as pool:
        raw = list(pool.map(both, PAIRS))
    calls = [w for pair in raw for w in pair]
    slot_a = calls.count("A") / len(calls)
    single_order_wins = sum(xy == "A" for xy, _ in raw)  # x "wins" if we only ask x-first
    verdicts = [swap_verdict(xy, yx) for xy, yx in raw]
    flips = verdicts.count("tie")
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=flips > 0 or abs(slot_a - 0.5) > 0.1,
        # Every declared winner on an equal pair is a false winner. Single order: all 12.
        fixed=verdicts.count("x") + verdicts.count("y") < len(PAIRS),
        evidence=f"{len(PAIRS)} equal-quality pairs, forced A/B, luna: slot A chosen "
                 f"{slot_a:.0%} of {len(calls)} calls; single order says x wins "
                 f"{single_order_wins}/{len(PAIRS)}; with swap: {flips} flipped -> tie, "
                 f"x {verdicts.count('x')}, y {verdicts.count('y')}",
        data={"slot_a_rate": slot_a, "flips": flips, "verdicts": verdicts, "raw": raw},
    )
