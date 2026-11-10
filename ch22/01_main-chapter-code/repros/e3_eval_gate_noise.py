"""E3: a fixed pass-rate bar on a small eval set mostly measures luck (simulation, no model)."""

from __future__ import annotations

import random

from ch22_common import Result, timed

ID, TITLE, LAYER, KIND = "E3", "Small eval gates are noisy", "Evals", "local"

# Per-item pass probabilities shaped like Chapter 20's 19-question data-bot gate on main:
# 17 solid items, s8 flaky (failed 1 of 3 repeats), j1 a stable known failure.
MAIN = {f"q{i}": 0.99 for i in range(17)} | {"s8": 0.67, "j1": 0.0}
BROKEN = MAIN | {"q0": 0.0}  # a prompt "cleanup" breaks one headline question (ch20's s1)


def sample(probs: dict[str, float], rng: random.Random, repeats: int = 1) -> dict[str, int]:
    """Number of passes per item over `repeats` runs."""
    return {k: sum(rng.random() < p for _ in range(repeats)) for k, p in probs.items()}


def bar_gate(passes: dict[str, int], repeats: int, bar: float) -> bool:
    """Naive gate: overall pass rate must reach the bar."""
    return sum(passes.values()) / (len(passes) * repeats) >= bar


def flip_gate(base: dict[str, int], cand: dict[str, int], repeats: int) -> bool:
    """Block only items that passed every repeat on main and failed every repeat on the PR."""
    return not any(base[k] == repeats and cand.get(k, 0) == 0 for k in base)


def rates(trials: int = 10_000, seed: int = 22) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    out = {}
    for name, cand_probs in (("unchanged PR", MAIN), ("PR breaks q0", BROKEN)):
        blocked = {"90% bar, 1 pass": 0, "95% bar, 1 pass": 0, "flip rule, 3 repeats": 0}
        for _ in range(trials):
            one = sample(cand_probs, rng)
            blocked["90% bar, 1 pass"] += not bar_gate(one, 1, 0.90)
            blocked["95% bar, 1 pass"] += not bar_gate(one, 1, 0.95)
            base, cand = sample(MAIN, rng, 3), sample(cand_probs, rng, 3)
            blocked["flip rule, 3 repeats"] += not flip_gate(base, cand, 3)
        out[name] = {k: v / trials for k, v in blocked.items()}
    return out


@timed
def run() -> Result:
    r = rates()
    false_block = r["unchanged PR"]
    catch = r["PR breaks q0"]
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=false_block["90% bar, 1 pass"] > 0.2,
        fixed=false_block["flip rule, 3 repeats"] < 0.05 and catch["flip rule, 3 repeats"] > 0.95,
        evidence="10,000 simulated gate runs, 19 items: blocks an unchanged PR " + ", ".join(
            f"{k} {v:.1%}" for k, v in false_block.items()) + "; catches a broken item " + ", ".join(
            f"{k} {v:.1%}" for k, v in catch.items()),
        data=r,
    )
