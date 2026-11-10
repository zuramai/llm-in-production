"""LoRA fine-tuning of an open-weights model with SageMaker AI serverless model customization.

This is the managed version of Chapter 18: same chat JSONL, same LoRA, but AWS picks and runs
the GPU. The adapter ends up in your account (a SageMaker model package backed by S3), so you
can serve it on a SageMaker endpoint, import it into Bedrock, or download it and serve it with
vLLM as in Chapter 18.

Needs an AWS account, credentials, an S3 bucket in the job's region, and an execution role
with the ``AmazonSageMakerModelCustomizationCoreAccess`` policy. SDK v3 (``sagemaker>=3``):
``Estimator`` and ``JumpStartEstimator`` are gone; ``sagemaker.train.SFTTrainer`` replaces them
for fine-tuning JumpStart hub models such as ``huggingface-reasoning-qwen3-8b``.

    cd ch19/01_main-chapter-code
    export AWS_DEFAULT_REGION=us-east-1     # the SDK creates a boto3 client at import time
    .venv-cloud/bin/python -m ch19_ft.sagemaker_ft train --bucket my-bucket --dry-run
    .venv-cloud/bin/python -m ch19_ft.sagemaker_ft train --bucket my-bucket
    .venv-cloud/bin/python -m ch19_ft.sagemaker_ft status
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ch19_ft import FORMATS_DIR, RESULTS
from ch19_ft.convert import validate_file

JOB_FILE = RESULTS / "sagemaker_job.json"
DEFAULT_MODEL = "huggingface-reasoning-qwen3-8b"


def upload(bucket: str, prefix: str, files: list[Path], region: str) -> list[str]:
    import boto3

    s3 = boto3.client("s3", region_name=region)
    uris = []
    for path in files:
        key = f"{prefix}/{path.name}"
        s3.upload_file(str(path), bucket, key)
        uris.append(f"s3://{bucket}/{key}")
    return uris


def cmd_train(args: argparse.Namespace) -> None:
    # The serverless SFT recipe takes OpenAI-style {"messages": [...]} rows with the system
    # message first, which is exactly formats/openai/.
    files = [args.dir / "train.jsonl", args.dir / "val.jsonl"]
    for rep in (validate_file("openai", f) for f in files):
        if not rep.ok:
            raise SystemExit(f"{rep.split}: {rep.errors[0]}")
    os.environ.setdefault("AWS_DEFAULT_REGION", args.region)
    from sagemaker.train import SFTTrainer
    from sagemaker.train.common import TrainingType

    train_uri, val_uri = upload(args.bucket, args.prefix, files, args.region)
    trainer = SFTTrainer(
        model=args.model,
        training_type=TrainingType.LORA,
        model_package_group=args.package_group,     # where the tuned adapter is registered
        training_dataset=train_uri,
        validation_dataset=val_uri,
        s3_output_path=f"s3://{args.bucket}/{args.prefix}/output/",
        accept_eula=True,                           # the base model's licence
        # no ``compute=``: serverless. Pass TrainingJobCompute(instance_type="ml.g5.2xlarge",
        # instance_count=1) to choose (and pay for) the instance yourself.
    )
    trainer.hyperparameters.max_epochs = args.epochs
    if args.dry_run:
        trainer.train(dry_run=True)  # checks IAM, data paths and hyperparameters, no job
        print("dry run passed")
        return
    trainer.train(wait=False)
    job = trainer.latest_training_job
    RESULTS.mkdir(exist_ok=True)
    JOB_FILE.write_text(json.dumps({"training_job_name": job.training_job_name,
                                    "region": args.region}, indent=2))
    print(f"started {job.training_job_name}; logs follow (Ctrl-C does not stop the job)")
    trainer.stream_logs()


def cmd_status(args: argparse.Namespace) -> None:
    saved = json.loads(JOB_FILE.read_text())
    os.environ.setdefault("AWS_DEFAULT_REGION", saved["region"])
    from sagemaker.core.resources import TrainingJob

    job = TrainingJob.get(training_job_name=saved["training_job_name"], region=saved["region"])
    saved.update(status=job.training_job_status, secondary=job.secondary_status,
                 model_package=job.output_model_package_arn)
    JOB_FILE.write_text(json.dumps(saved, indent=2, default=str))
    print(json.dumps(saved, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train")
    train.add_argument("--bucket", required=True)
    train.add_argument("--prefix", default="ch19-triage")
    train.add_argument("--region", default="us-east-1")
    train.add_argument("--model", default=DEFAULT_MODEL)
    train.add_argument("--package-group", default="shopflow-triage")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--dir", type=Path, default=FORMATS_DIR / "openai")
    train.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    args = ap.parse_args()
    {"train": cmd_train, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
