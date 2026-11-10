"""Supervised tuning of Gemini on Vertex AI (Gemini Enterprise Agent Platform), via google-genai.

Needs a GCP project with the Vertex AI API enabled, a Cloud Storage bucket, and application
default credentials (``gcloud auth application-default login``). Cloud SDKs are not in the book's
main environment:

    uv venv ch19/01_main-chapter-code/.venv-cloud --python 3.12
    UV_CACHE_DIR=$HOME/.cache/uv-cloud-ch19 uv pip install \
        -r ch19/01_main-chapter-code/requirements-cloud.txt \
        --python ch19/01_main-chapter-code/.venv-cloud
    cd ch19/01_main-chapter-code
    .venv-cloud/bin/python -m ch19_ft.vertex_ft tune --project my-proj --bucket my-bucket
    .venv-cloud/bin/python -m ch19_ft.vertex_ft poll --project my-proj
    .venv-cloud/bin/python -m ch19_ft.vertex_ft eval --project my-proj --limit 50

``vertexai.tuning.sft`` (google-cloud-aiplatform) is deprecated; ``client.tunings`` in the Google
Gen AI SDK replaces it. Gemini 3.x models tune only on the ``us`` and ``eu`` multi-region
endpoints, and a project runs one tuning job at a time unless you raise the quota.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ch19_ft import DATA, FORMATS_DIR, RESULTS, previous
from ch19_ft.convert import validate_file

JOB_FILE = RESULTS / "vertex_job.json"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
RUNNING = ("JOB_STATE_PENDING", "JOB_STATE_QUEUED", "JOB_STATE_RUNNING")


def client(project: str, location: str) -> Any:
    from google import genai
    from google.genai import types

    # vertexai=True: without it the client silently targets the Gemini Developer API, which has
    # no tuning (and rejects gcs_uri).
    return genai.Client(vertexai=True, project=project, location=location,
                        http_options=types.HttpOptions(api_version="v1beta1"))


def upload(project: str, bucket: str, prefix: str, files: list[Path]) -> list[str]:
    """Copy the JSONL files to gs://bucket/prefix/ (google-genai has no upload call)."""
    from google.cloud import storage

    b = storage.Client(project=project).bucket(bucket)
    uris = []
    for path in files:
        blob = b.blob(f"{prefix}/{path.name}")
        blob.upload_from_filename(str(path), content_type="application/jsonl")
        uris.append(f"gs://{bucket}/{blob.name}")
    return uris


def tune_config(display_name: str, epochs: int, adapter_size: str, val_uri: str | None) -> Any:
    from google.genai import types

    return types.CreateTuningJobConfig(
        tuned_model_display_name=display_name,
        epoch_count=epochs,
        adapter_size=adapter_size,
        # Not TuningDataset: the SDK only prints "Type mismatch" for that, it doesn't raise.
        validation_dataset=types.TuningValidationDataset(gcs_uri=val_uri) if val_uri else None,
        labels={"book": "llm-in-production", "chapter": "19"},
    )


def cmd_tune(args: argparse.Namespace) -> None:
    from google.genai import types

    files = [args.dir / "train.jsonl", args.dir / "val.jsonl"]
    for rep in (validate_file("vertex", f) for f in files):
        if not rep.ok:
            raise SystemExit(f"{rep.split}: {rep.errors[0]}")
    train_uri, val_uri = upload(args.project, args.bucket, args.prefix, files)
    job = client(args.project, args.location).tunings.tune(
        base_model=args.model,
        training_dataset=types.TuningDataset(gcs_uri=train_uri),
        config=tune_config("shopflow-triage", args.epochs, args.adapter_size, val_uri),
    )
    RESULTS.mkdir(exist_ok=True)
    JOB_FILE.write_text(json.dumps({"name": job.name, "state": str(job.state),
                                    "location": args.location}, indent=2))
    print(f"created {job.name} ({job.state})")


def cmd_poll(args: argparse.Namespace) -> None:
    saved = json.loads(JOB_FILE.read_text())
    c = client(args.project, saved["location"])
    job = c.tunings.get(name=saved["name"])
    while str(job.state).split(".")[-1] in RUNNING:  # the enum prints as JobState.JOB_STATE_...
        print(f"{time.strftime('%H:%M:%S')} {job.state}")
        time.sleep(args.interval)
        job = c.tunings.get(name=saved["name"])  # no refresh(): get() returns a new object
    saved.update(state=str(job.state), error=job.error and job.error.message)
    if job.tuned_model:
        saved.update(model=job.tuned_model.model, endpoint=job.tuned_model.endpoint)
        for cp in job.tuned_model.checkpoints or []:
            print(f"checkpoint {cp.checkpoint_id} epoch={cp.epoch} step={cp.step}")
    JOB_FILE.write_text(json.dumps(saved, indent=2))
    print(json.dumps(saved, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    """The 210-ticket test set on the tuned endpoint, scored with Chapter 2's ``score``.

    No ``response_schema``: Google's tuning docs warn that controlled generation on a tuned
    model can lower quality, so we parse the JSON the model learned to write.
    """
    from google.genai import types

    from ch19_ft.evaluate import as_ticket, segment_scores

    p = previous()
    saved = json.loads(JOB_FILE.read_text())
    c = client(args.project, saved["location"])
    test = p.read_jsonl(DATA / "test.jsonl")[:args.limit]
    config = types.GenerateContentConfig(
        system_instruction=p.FT_RULES + p.FORMAT_HINT, max_output_tokens=300,
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
    )
    results = []
    for rec in test:
        t = as_ticket(rec)
        res = p.Result(t.id, "vertex", saved["endpoint"], None, 1, False, False, None)
        t0 = time.perf_counter()
        resp = c.models.generate_content(
            model=saved["endpoint"], config=config,
            contents=p.format_ticket(t.subject, t.body, t.channel))
        res.latency_ms = round((time.perf_counter() - t0) * 1000)
        res.raw = resp.text or ""
        try:
            res.triage = p.parse_triage(res.raw).model_dump()
            res.first_try_ok = res.first_try_json_ok = True
        except p.TriageError as e:
            res.final_error = e.kind
            res.errors.append(f"{e.kind}: {e}")
        usage = resp.usage_metadata
        res.prompt_tokens = usage.prompt_token_count or 0
        res.completion_tokens = usage.candidates_token_count or 0
        results.append(res)
    scores = segment_scores(results, test, p.read_jsonl(DATA / "train.jsonl"))
    out = {"endpoint": saved["endpoint"], "n": len(test), "scores": scores,
           "results": [r.to_json() for r in results]}
    (RESULTS / "eval__vertex__tuned.json").write_text(json.dumps(out, indent=0))
    for seg, s in scores.items():
        print(f"{seg:7} n={s['n']:3} category={s['category_acc_%']}% "
              f"priority={s['priority_acc_%']}% valid={s['final_valid_%']}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tune = sub.add_parser("tune")
    tune.add_argument("--bucket", required=True)
    tune.add_argument("--prefix", default="ch19-triage")
    tune.add_argument("--location", default="us", help="Gemini 3.x: 'us' or 'eu' only")
    tune.add_argument("--model", default=DEFAULT_MODEL)
    tune.add_argument("--epochs", type=int, default=3)
    tune.add_argument("--adapter-size", default="ADAPTER_SIZE_FOUR")
    tune.add_argument("--dir", type=Path, default=FORMATS_DIR / "vertex")
    poll = sub.add_parser("poll")
    poll.add_argument("--interval", type=int, default=60)
    ev = sub.add_parser("eval")
    ev.add_argument("--limit", type=int)
    args = ap.parse_args()
    {"tune": cmd_tune, "poll": cmd_poll, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
