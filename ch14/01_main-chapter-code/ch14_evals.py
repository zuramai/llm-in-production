"""Langfuse datasets and experiments for the data bot, scored with Chapter 3/6 evals.

    uv run python ch14/01_main-chapter-code/ch14_evals.py --run-name luna-planner-v1
    uv run python ch14/01_main-chapter-code/ch14_evals.py --run-name luna-planner-v2 \\
        --prompt-label candidate

The 30 Chapter 6 questions become a Langfuse *dataset*. Each run of the bot over it is an
*experiment* (a "dataset run"): one trace per item, linked to the item, with scores attached.
The scorers are plain evalkit scorers, `scorer(case, output) -> float`, the interface from
Chapter 3, so the same function scores a local `evalkit.run(...)` and a Langfuse experiment.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import ch14_obs as obs
import openai
import psycopg
import yaml
from ch14_sqlbot import answer
from previous_chapters import Case, Result, gold_rows, judge, rows_match

from llmprod import chat_client, get_settings

HERE = Path(__file__).resolve().parent

DATASET = "sqlbot-eval"
OLLAMA_URL = "http://localhost:11434/v1"
ANSWERABLE = ("simple", "join", "time_window")


def load_cases(tz: str, path: Path = HERE / "eval" / "questions.yaml") -> list[Case]:
    """Chapter 6's eval set as evalkit Cases: gold SQL -> metric "sql", else metric "judge"."""
    cases = []
    for q in yaml.safe_load(path.read_text().replace("{tz}", tz)):
        if "gold_sql" in q:
            cases.append(Case(q["id"], q["question"], q["gold_sql"].strip(), "sql",
                              meta={"category": q["category"]}))
        else:
            cases.append(Case(q["id"], q["question"], q["expected_behavior"], "judge",
                              meta={"category": q["category"]}))
    return cases


def case_to_item(case: Case) -> dict:
    """evalkit Case -> Langfuse dataset item fields."""
    key = "gold_sql" if case.metric == "sql" else "expected_behavior"
    return {"id": f"{DATASET}-{case.id}", "input": {"question": case.input},
            "expected_output": {key: case.expected},
            "metadata": {"case_id": case.id, "category": case.meta.get("category")}}


def item_to_case(input: dict, expected_output: dict, metadata: dict | None) -> Case:
    """Langfuse dataset item fields -> evalkit Case (the inverse of case_to_item)."""
    metadata = metadata or {}
    meta = {"category": metadata.get("category")}
    if "gold_sql" in expected_output:
        return Case(metadata.get("case_id", ""), input["question"],
                    expected_output["gold_sql"], "sql", meta=meta)
    return Case(metadata.get("case_id", ""), input["question"],
                expected_output["expected_behavior"], "judge", meta=meta)


def sync_dataset(lf, cases: list[Case]) -> int:
    """Create the dataset and upsert one item per case (stable ids make re-runs idempotent)."""
    lf.create_dataset(name=DATASET, description="ShopFlow data bot: Chapter 6 eval questions",
                      metadata={"source": "ch06/01_main-chapter-code/eval/questions.yaml"})
    for case in cases:
        lf.create_dataset_item(dataset_name=DATASET, **case_to_item(case))
    return len(cases)


# --------------------------------------------------------------------------- scorers (evalkit)

def execution_accuracy(case: Case, output: dict) -> float:
    """1.0 if the bot's rows contain the gold rows (Chapter 6 execution accuracy)."""
    return float(rows_match(gold_rows(case.expected), output.get("rows") if output.get("sql")
                            else None))


def make_behavior_scorer(judge_model: str, client=None) -> Callable[[Case, dict], float]:
    def behavior(case: Case, output: dict) -> float:
        transcript = json.dumps({k: output.get(k) for k in ("answer", "sql", "error")}
                                | {"rows_preview": (output.get("rows") or [])[:5]}, default=str)
        verdict = judge(case.input, case.expected, f"Bot output:\n{transcript}", judge_model,
                        client=client)
        output.setdefault("_judge", verdict)
        return float(verdict.get("score") == 2)
    return behavior


def make_correct_scorer(judge_model: str, client=None) -> Callable[[Case, dict], float]:
    """Dispatch on the case's metric, like Chapter 6's `score()`."""
    behavior = make_behavior_scorer(judge_model, client)
    return lambda case, output: (execution_accuracy(case, output) if case.metric == "sql"
                                 else behavior(case, output))


def as_langfuse_evaluator(name: str, scorer: Callable[[Case, dict], float]):
    """Adapt an evalkit scorer to Langfuse's evaluator signature."""
    from langfuse import Evaluation

    def evaluator(*, input, output, expected_output=None, metadata=None, **_):
        case = item_to_case(input, expected_output or {}, metadata)
        if output.get("error"):
            return Evaluation(name=name, value=0.0, comment=output["error"][:500])
        value = float(scorer(case, output))
        reason = (output.get("_judge") or {}).get("reason")
        if reason is None and value < 1:
            # Execution-accuracy miss: keep what the bot did, so the UI shows why it failed.
            reason = (f"rows={json.dumps((output.get('rows') or [])[:2], default=str)} "
                      f"answer={output.get('answer')!r} sql={output.get('sql')}")[:500]
        return Evaluation(name=name, value=value, comment=reason,
                          metadata={"category": case.meta.get("category")})
    return evaluator


def accuracy_by_group(*, item_results, **_):
    """Run-level scores: overall, execution (answerable) and behavior accuracy."""
    from langfuse import Evaluation

    groups: dict[str, list[float]] = {"accuracy": [], "execution_accuracy": [],
                                      "behavior_accuracy": []}
    for r in item_results:
        value = next((e.value for e in r.evaluations if e.name == "correct"), 0.0)
        groups["accuracy"].append(value)
        exec_item = (r.item.metadata or {}).get("category") in ANSWERABLE
        groups["execution_accuracy" if exec_item else "behavior_accuracy"].append(value)
    return [Evaluation(name=k, value=sum(v) / len(v)) for k, v in groups.items() if v]


def to_evalkit_results(experiment) -> list[Result]:
    """Langfuse ExperimentResult -> evalkit Results, so Chapter 3's `summarize` works on it."""
    out = []
    for r in experiment.item_results:
        case = item_to_case(r.item.input, r.item.expected_output, r.item.metadata)
        score = next((e.value for e in r.evaluations if e.name == "correct"), None)
        out.append(Result(case, (r.output or {}).get("answer") or "", score,
                          (r.output or {}).get("latency_ms", 0) / 1000,
                          (r.output or {}).get("error")))
    return out


def db_fingerprint() -> tuple:
    with psycopg.connect(get_settings().database_url) as conn:
        return conn.execute(
            "SELECT (SELECT count(*) FROM orders), (SELECT sum(total_cents) FROM orders)"
        ).fetchone()


def make_task(model: str, prompt_label: str, client: openai.OpenAI,
              reasoning_effort: str | None, lf):
    def task(*, item, **_):
        try:
            res = answer(item.input["question"], model=model, client=client,
                         reasoning_effort=reasoning_effort, prompt_label=prompt_label,
                         user_id="eval-bot", tags=["experiment"], langfuse=lf)
        except Exception as e:  # noqa: BLE001 — a crash is a scored failure, not a stopped run
            return {"answer": None, "sql": None, "rows": None,
                    "error": f"{type(e).__name__}: {e}"}
        return {"answer": res.answer, "sql": res.sql, "rows": res.rows[:50],
                "row_count": len(res.rows), "attempts": res.attempts,
                "latency_ms": res.latency_ms, "cost_usd": res.usage.get("cost_usd", 0.0),
                "tokens": res.usage.get("total_tokens", 0), "prompt_versions":
                res.prompt_versions}
    return task


def run(run_name: str, *, model: str | None = None, prompt_label: str = "production",
        provider: str = "default", reasoning_effort: str | None = None,
        judge_model: str | None = None, concurrency: int | None = None,
        only: set[str] | None = None, lf=None):
    s = get_settings()
    lf = lf or obs.client()
    model = model or s.llm_model
    client = (openai.OpenAI(base_url=OLLAMA_URL, api_key="ollama", timeout=600)
              if provider == "ollama" else chat_client(timeout=120))
    sync_dataset(lf, load_cases(s.business_timezone))
    dataset = lf.get_dataset(DATASET)
    items = [i for i in dataset.items
             if not only or (i.metadata or {}).get("case_id") in only]
    before = db_fingerprint()
    result = lf.run_experiment(
        name=DATASET, run_name=run_name, data=items,
        description=f"{model} · planner label {prompt_label}",
        task=make_task(model, prompt_label, client, reasoning_effort, lf),
        evaluators=[as_langfuse_evaluator("correct", make_correct_scorer(
            judge_model or s.llm_model_strong))],
        run_evaluators=[accuracy_by_group],
        max_concurrency=concurrency or (1 if provider == "ollama" else 6),
        metadata={"model": model, "prompt_label": prompt_label, "provider": provider},
    )
    lf.flush()
    return result, before == db_fingerprint()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-name", required=True)
    p.add_argument("--model")
    p.add_argument("--prompt-label", default="production")
    p.add_argument("--provider", choices=["default", "ollama"], default="default")
    p.add_argument("--reasoning-effort")
    p.add_argument("--only", help="comma-separated case ids")
    p.add_argument("--out", type=Path, help="write run scores + per-item results as JSON")
    args = p.parse_args()
    from previous_chapters import summarize

    result, db_unchanged = run(args.run_name, model=args.model, prompt_label=args.prompt_label,
                               provider=args.provider, reasoning_effort=args.reasoning_effort,
                               only=set(args.only.split(",")) if args.only else None)
    run_scores = {e.name: round(e.value, 3) for e in result.run_evaluations}
    summary = summarize(to_evalkit_results(result))
    print(run_scores)
    print(summary, "| db unchanged:", db_unchanged)
    print(result.experiment_url)
    if args.out:
        items = [{"id": r.item.metadata["case_id"], "category": r.item.metadata["category"],
                  "correct": next((e.value for e in r.evaluations if e.name == "correct"), 0.0),
                  "latency_ms": (r.output or {}).get("latency_ms"),
                  "tokens": (r.output or {}).get("tokens", 0),
                  "cost_usd": (r.output or {}).get("cost_usd", 0.0)}
                 for r in result.item_results]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"run_name": args.run_name, "model": args.model,
                                        "provider": args.provider,
                                        "prompt_label": args.prompt_label,
                                        "reasoning_effort": args.reasoning_effort,
                                        "run_scores": run_scores, "summary": summary,
                                        "db_unchanged": db_unchanged, "items": items},
                                       indent=1, default=str))


if __name__ == "__main__":
    main()
