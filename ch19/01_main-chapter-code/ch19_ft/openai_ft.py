"""Fine-tune on OpenAI: upload, launch, poll, evaluate. Needs ``OPENAI_API_KEY`` in ``.env``.

    cd ch19/01_main-chapter-code
    uv run python -m ch19_ft.convert                          # formats/openai/{train,val}.jsonl
    uv run python -m ch19_ft.openai_ft launch --dry-run       # checks + cost estimate, no API call
    uv run python -m ch19_ft.openai_ft launch                 # uploads and creates the job
    uv run python -m ch19_ft.openai_ft poll                   # waits, prints events, saves model id
    uv run python -m ch19_ft.openai_ft eval                   # the 210-ticket test set on it

    # the same eval on any OpenRouter model (the "before" number; uses LLM_API_KEY)
    uv run python -m ch19_ft.openai_ft eval --provider openrouter --model openai/gpt-6-luna

As of 2026-11 OpenAI is winding down self-serve fine-tuning: only organisations that ran
inference on a fine-tuned model in the last 60 days can create jobs, nobody can after
2027-01-06, and the tunable base models (and every model tuned from them) shut down on
2027-01-14. ``launch`` prints that date for the model you pick. OpenRouter has no fine-tuning
API, so this script talks to api.openai.com directly.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

from ch19_ft import FORMATS_DIR, HERE, RESULTS, previous
from ch19_ft.convert import validate_file
from ch19_ft.pricing import OPENAI_FT, training_cost

JOB_FILE = RESULTS / "openai_job.json"
TERMINAL = ("succeeded", "failed", "cancelled")
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"


def openai_key() -> str:
    """``OPENAI_API_KEY`` from the environment or the repo's ``.env``; a clear error if absent."""
    from dotenv import load_dotenv

    load_dotenv(HERE.parents[1] / ".env")
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set in .env. Fine-tuning needs a direct OpenAI key: "
            "OpenRouter (LLM_API_KEY) has no fine-tuning API."
        )
    return key


def plan(model: str, epochs: int, train: Path, val: Path | None,
         today: dt.date | None = None) -> dict[str, Any]:
    """Validate the files and price the job before anything is uploaded."""
    if model not in OPENAI_FT:
        raise SystemExit(f"{model!r} is not fine-tunable; choose one of {sorted(OPENAI_FT)}")
    reports = [validate_file("openai", p) for p in (train, val) if p is not None]
    for rep in reports:
        if not rep.ok:
            first = rep.errors[0]
            raise SystemExit(f"{rep.split}: {len(rep.errors)} errors, first on line "
                             f"{first.line}: {first.code}: {first.message}")
    price = OPENAI_FT[model]
    tokens = sum(reports[0].tokens)
    today = today or dt.date.today()
    return {
        "model": model, "epochs": epochs,
        "train_examples": reports[0].n, "val_examples": reports[1].n if val else 0,
        "train_tokens_per_epoch": tokens,
        "estimated_training_usd": round(training_cost(tokens, epochs, price.training_per_m), 2),
        "shutdown": price.shutdown.isoformat(),
        "days_of_service_left": (price.shutdown - today).days,
    }


def job_request(model: str, epochs: int, train_id: str, val_id: str | None,
                suffix: str = "shopflow-triage", seed: int = 19) -> dict[str, Any]:
    """Keyword arguments for ``client.fine_tuning.jobs.create`` (method-style hyperparameters)."""
    req: dict[str, Any] = {
        "model": model, "training_file": train_id, "suffix": suffix, "seed": seed,
        "method": {"type": "supervised",
                   "supervised": {"hyperparameters": {"n_epochs": epochs}}},
        "metadata": {"book": "llm-in-production", "chapter": "19"},
    }
    if val_id:
        req["validation_file"] = val_id
    return req


def new_events(events: list[Any], seen: set[str]) -> list[Any]:
    """Events not printed yet, oldest first (the API lists newest first)."""
    fresh = [e for e in events if e.id not in seen]
    seen.update(e.id for e in fresh)
    return sorted(fresh, key=lambda e: e.created_at)


def cmd_launch(args: argparse.Namespace) -> None:
    train, val = args.dir / "train.jsonl", args.dir / "val.jsonl"
    p = plan(args.model, args.epochs, train, val if val.exists() else None)
    print(json.dumps(p, indent=2))
    if p["days_of_service_left"] < 90:
        print(f"WARNING: {args.model} and every model tuned from it shut down on "
              f"{p['shutdown']} ({p['days_of_service_left']} days from today).")
    if p["estimated_training_usd"] > args.max_usd:
        raise SystemExit(f"estimated ${p['estimated_training_usd']} > --max-usd {args.max_usd}")
    if args.dry_run:
        print(json.dumps(job_request(args.model, args.epochs, "file-<train>",
                                     "file-<val>" if val.exists() else None), indent=2))
        return
    import openai

    client = openai.OpenAI(api_key=openai_key())
    ids = {}
    for name, path in (("train", train), ("val", val)):
        if path.exists():
            f = client.files.create(file=path, purpose="fine-tune")
            ids[name] = client.files.wait_for_processing(f.id).id
    try:
        job = client.fine_tuning.jobs.create(**job_request(args.model, args.epochs, ids["train"],
                                                           ids.get("val")))
    except openai.APIStatusError as e:
        raise SystemExit(f"OpenAI refused the job ({e.status_code}): {e.message}\nSince "
                         "2026-07-02 only organisations that ran inference on a fine-tuned "
                         "model in the last 60 days can create jobs.") from e
    RESULTS.mkdir(exist_ok=True)
    JOB_FILE.write_text(json.dumps({"job_id": job.id, "plan": p, "files": ids}, indent=2))
    print(f"created {job.id} ({job.status}); next: python -m ch19_ft.openai_ft poll")


def cmd_poll(args: argparse.Namespace) -> None:
    import openai

    client = openai.OpenAI(api_key=openai_key())
    saved = json.loads(JOB_FILE.read_text())
    job_id = args.job or saved["job_id"]
    seen: set[str] = set()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        events = client.fine_tuning.jobs.list_events(job_id, limit=50).data
        for e in new_events(events, seen):
            print(f"{dt.datetime.fromtimestamp(e.created_at):%H:%M:%S} {e.level:5} {e.message}")
        if job.status in TERMINAL:
            break
        time.sleep(args.interval)
    saved.update(status=job.status, fine_tuned_model=job.fine_tuned_model,
                 trained_tokens=job.trained_tokens, error=job.error and job.error.model_dump())
    JOB_FILE.write_text(json.dumps(saved, indent=2))
    for ck in client.fine_tuning.jobs.checkpoints.list(job_id).data:
        print(f"checkpoint step {ck.step_number}: {ck.fine_tuned_model_checkpoint} {ck.metrics}")
    print(f"{job.status}: {job.fine_tuned_model} trained_tokens={job.trained_tokens}")


def cmd_eval(args: argparse.Namespace) -> None:
    from ch19_ft.evaluate import evaluate

    p = previous()
    if args.provider == "openrouter":
        model, client, price = args.model, p.async_chat_client(timeout=120), None
    else:
        import openai

        model = args.model or json.loads(JOB_FILE.read_text())["fine_tuned_model"]
        base = model.split(":")[1] if model.startswith("ft:") else model
        ft = OPENAI_FT.get(base)
        price = (ft.input_per_m, ft.output_per_m) if ft else None
        client = openai.AsyncOpenAI(api_key=openai_key(), timeout=120, max_retries=2)
    tag = f"{args.provider}__{model.replace('/', '_').replace(':', '_')}__ft-prompt"
    out = asyncio.run(evaluate(client, model, tag, limit=args.limit, price=price,
                               concurrency=args.concurrency))
    for seg, s in out["scores"].items():
        print(f"{seg:7} n={s['n']:3} category={s['category_acc_%']}% "
              f"priority={s['priority_acc_%']}% order_id={s['order_id_acc_%']}% "
              f"valid={s['final_valid_%']}% $/1k={s['cost_per_1k_usd']} "
              f"p50={s['latency_p50_ms']} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--model", default=DEFAULT_MODEL)
    launch.add_argument("--epochs", type=int, default=3)
    launch.add_argument("--dir", type=Path, default=FORMATS_DIR / "openai")
    launch.add_argument("--max-usd", type=float, default=5.0)
    launch.add_argument("--dry-run", action="store_true")
    poll = sub.add_parser("poll")
    poll.add_argument("--job")
    poll.add_argument("--interval", type=int, default=60)
    ev = sub.add_parser("eval")
    ev.add_argument("--provider", choices=("openai", "openrouter"), default="openai")
    ev.add_argument("--model", help="default: the fine_tuned_model saved by poll")
    ev.add_argument("--limit", type=int)
    ev.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    {"launch": cmd_launch, "poll": cmd_poll, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
