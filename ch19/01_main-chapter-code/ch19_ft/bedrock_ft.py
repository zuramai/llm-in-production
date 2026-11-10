"""Fine-tune Amazon Nova on Bedrock, then serve it on demand (pay per token).

Needs an AWS account with Bedrock model access in us-east-1 (where Nova fine-tuning runs), an
S3 bucket in the same region, and an IAM role Bedrock can assume to read and write it.

    cd ch19/01_main-chapter-code
    .venv-cloud/bin/python -m ch19_ft.bedrock_ft train --bucket my-bucket \
        --role arn:aws:iam::123456789012:role/BedrockCustomization
    .venv-cloud/bin/python -m ch19_ft.bedrock_ft poll
    .venv-cloud/bin/python -m ch19_ft.bedrock_ft deploy       # on-demand custom model deployment
    .venv-cloud/bin/python -m ch19_ft.bedrock_ft ask "Where is my order #3190?"

There is no event stream: Bedrock writes per-epoch loss files to the output S3 prefix.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ch19_ft import FORMATS_DIR, RESULTS, previous
from ch19_ft.convert import validate_file

JOB_FILE = RESULTS / "bedrock_job.json"
BASE_MODELS = {"nova-micro": "amazon.nova-micro-v1:0:128k",
               "nova-lite": "amazon.nova-lite-v1:0:300k",
               "nova-pro": "amazon.nova-pro-v1:0:300k"}
RUNNING = ("InProgress", "Stopping")  # then Completed | Failed | Stopped (not "COMPLETE")


def hyperparameters(epochs: int, n_train: int, model: str) -> dict[str, str]:
    """Nova's documented knobs, as strings. Warm-up: dataset size / 640 (Micro), /160 (Lite),
    /320 (Pro), the user guide's rule of thumb."""
    if not 1 <= epochs <= 5:
        raise ValueError("Nova accepts 1 to 5 epochs")
    per = {"nova-micro": 640, "nova-lite": 160, "nova-pro": 320}[model]
    return {"epochCount": str(epochs), "learningRate": "0.00001",
            "learningRateWarmupSteps": str(max(0, min(100, round(n_train / per))))}


def cmd_train(args: argparse.Namespace) -> None:
    import boto3

    files = [args.dir / "train.jsonl", args.dir / "val.jsonl"]
    reports = [validate_file("bedrock", f) for f in files]
    for rep in reports:
        if not rep.ok:
            raise SystemExit(f"{rep.split}: {rep.errors[0]}")
    s3 = boto3.client("s3", region_name=args.region)
    uris = []
    for f in files:
        key = f"{args.prefix}/{f.name}"
        s3.upload_file(str(f), args.bucket, key)
        uris.append(f"s3://{args.bucket}/{key}")  # a file URI: a prefix is rejected
    name = f"shopflow-triage-{time.strftime('%Y%m%d-%H%M%S')}"
    bedrock = boto3.client("bedrock", region_name=args.region)
    job = bedrock.create_model_customization_job(
        jobName=name, customModelName=name, roleArn=args.role,
        baseModelIdentifier=BASE_MODELS[args.model], customizationType="FINE_TUNING",
        trainingDataConfig={"s3Uri": uris[0]},
        validationDataConfig={"validators": [{"s3Uri": uris[1]}]},
        outputDataConfig={"s3Uri": f"s3://{args.bucket}/{args.prefix}/output/"},
        hyperParameters=hyperparameters(args.epochs, reports[0].n, args.model),
        jobTags=[{"key": "book", "value": "llm-in-production-ch19"}],
    )
    RESULTS.mkdir(exist_ok=True)
    JOB_FILE.write_text(json.dumps({"jobArn": job["jobArn"], "region": args.region}, indent=2))
    print(f"created {job['jobArn']}")


def cmd_poll(args: argparse.Namespace) -> None:
    import boto3

    saved = json.loads(JOB_FILE.read_text())
    bedrock = boto3.client("bedrock", region_name=saved["region"])
    while True:
        job = bedrock.get_model_customization_job(jobIdentifier=saved["jobArn"])
        phases = {k: v.get("status") for k, v in (job.get("statusDetails") or {}).items()}
        print(f"{time.strftime('%H:%M:%S')} {job['status']} {phases}")
        if job["status"] not in RUNNING:
            break
        time.sleep(args.interval)
    saved.update(status=job["status"], outputModelArn=job.get("outputModelArn"),
                 failureMessage=job.get("failureMessage"),
                 trainingMetrics=job.get("trainingMetrics"))
    JOB_FILE.write_text(json.dumps(saved, indent=2, default=str))


def cmd_deploy(args: argparse.Namespace) -> None:
    """On-demand deployment: per-token pricing, no Provisioned Throughput commitment."""
    import boto3

    saved = json.loads(JOB_FILE.read_text())
    bedrock = boto3.client("bedrock", region_name=saved["region"])
    dep = bedrock.create_custom_model_deployment(modelDeploymentName="shopflow-triage-od",
                                                 modelArn=saved["outputModelArn"])
    arn = dep["customModelDeploymentArn"]
    while (status := bedrock.get_custom_model_deployment(
            customModelDeploymentIdentifier=arn)["status"]) == "Creating":
        time.sleep(30)
    saved.update(deploymentArn=arn, deploymentStatus=status)
    JOB_FILE.write_text(json.dumps(saved, indent=2, default=str))
    print(f"{status}: {arn}")


def cmd_ask(args: argparse.Namespace) -> None:
    """One ticket through the Converse API, with the *same* system prompt as training."""
    import boto3

    p = previous()
    saved = json.loads(JOB_FILE.read_text())
    rt = boto3.client("bedrock-runtime", region_name=saved["region"])
    resp = rt.converse(
        modelId=saved["deploymentArn"],
        system=[{"text": p.FT_RULES + p.FORMAT_HINT}],
        messages=[{"role": "user", "content": [{"text": p.format_ticket(
            args.subject, args.text, args.channel)}]}],
        inferenceConfig={"maxTokens": 300, "temperature": 0},
    )
    print(resp["output"]["message"]["content"][0]["text"])
    print(resp["usage"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train")
    train.add_argument("--bucket", required=True)
    train.add_argument("--role", required=True)
    train.add_argument("--prefix", default="ch19-triage")
    train.add_argument("--region", default="us-east-1")
    train.add_argument("--model", choices=sorted(BASE_MODELS), default="nova-micro")
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--dir", type=Path, default=FORMATS_DIR / "bedrock")
    poll = sub.add_parser("poll")
    poll.add_argument("--interval", type=int, default=60)
    sub.add_parser("deploy")
    ask = sub.add_parser("ask")
    ask.add_argument("text")
    ask.add_argument("--subject", default="(no subject)")
    ask.add_argument("--channel", default="email")
    args = ap.parse_args()
    {"train": cmd_train, "poll": cmd_poll, "deploy": cmd_deploy, "ask": cmd_ask}[args.cmd](args)


if __name__ == "__main__":
    main()
