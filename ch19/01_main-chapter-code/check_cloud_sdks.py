"""Offline check of the cloud launchers against the real SDKs (no account, no network calls).

    cd ch19/01_main-chapter-code
    AWS_DEFAULT_REGION=us-east-1 .venv-cloud/bin/python check_cloud_sdks.py

Compiling a script proves the syntax. This also proves the names: every boto3 operation and
parameter we call exists in botocore's service model, the google-genai config objects we build
validate, the SageMaker SFTTrainer accepts our keyword arguments, and the OpenAI job request
matches the SDK's signature.
"""

from __future__ import annotations

import importlib.metadata as md
import inspect
import py_compile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["convert", "openai_ft", "bedrock_ft", "vertex_ft", "sagemaker_ft"]

# (service, operation, parameters we pass) as used in ch19_ft/bedrock_ft.py
BOTO_CALLS = [
    ("bedrock", "CreateModelCustomizationJob",
     ["jobName", "customModelName", "roleArn", "baseModelIdentifier", "customizationType",
      "trainingDataConfig", "validationDataConfig", "outputDataConfig", "hyperParameters",
      "jobTags"]),
    ("bedrock", "GetModelCustomizationJob", ["jobIdentifier"]),
    ("bedrock", "CreateCustomModelDeployment", ["modelDeploymentName", "modelArn"]),
    ("bedrock", "GetCustomModelDeployment", ["customModelDeploymentIdentifier"]),
    ("bedrock-runtime", "Converse", ["modelId", "system", "messages", "inferenceConfig"]),
]


def check_boto() -> None:
    import botocore.session

    session = botocore.session.get_session()
    for service, op, params in BOTO_CALLS:
        model = session.get_service_model(service).operation_model(op)
        members = set(model.input_shape.members)
        missing = [p for p in params if p not in members]
        assert not missing, f"{service}.{op}: unknown parameters {missing}"
        print(f"ok  boto3 {service}.{op}({', '.join(params)})")
    statuses = (session.get_service_model("bedrock").shape_for("ModelCustomizationJobStatus")
                .enum)
    print(f"ok  bedrock job statuses: {statuses}")


def check_genai() -> None:
    from ch19_ft.vertex_ft import RUNNING, tune_config
    from google.genai import types
    from google.genai.tunings import Tunings

    cfg = tune_config("shopflow-triage", 3, "ADAPTER_SIZE_FOUR", "gs://bucket/val.jsonl")
    # google-genai only *prints* "Type mismatch" for a wrong nested type, so check it ourselves.
    assert isinstance(cfg.validation_dataset, types.TuningValidationDataset), type(
        cfg.validation_dataset)
    assert cfg.adapter_size == types.AdapterSize.ADAPTER_SIZE_FOUR
    types.GenerateContentConfig(system_instruction="x", max_output_tokens=300,
                                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"))
    assert all(hasattr(Tunings, m) for m in ("tune", "get", "list", "cancel"))
    states = {s.value for s in types.JobState}
    assert set(RUNNING) <= states, set(RUNNING) - states
    print(f"ok  google-genai CreateTuningJobConfig(epoch_count={cfg.epoch_count}, "
          f"adapter_size={cfg.adapter_size.value}, validation_dataset="
          f"{type(cfg.validation_dataset).__name__}); Tunings.tune/get; states {sorted(RUNNING)}")


def check_sagemaker() -> None:
    from sagemaker.train import SFTTrainer
    from sagemaker.train.common import TrainingType

    params = inspect.signature(SFTTrainer.__init__).parameters
    used = ["model", "training_type", "model_package_group", "training_dataset",
            "validation_dataset", "s3_output_path", "accept_eula"]
    assert all(p in params for p in used), [p for p in used if p not in params]
    assert "dry_run" in inspect.signature(SFTTrainer.train).parameters
    assert TrainingType.LORA.value == "LORA"
    print(f"ok  sagemaker SFTTrainer({', '.join(used)}).train(dry_run=...)")


def check_openai() -> None:
    import openai
    from ch19_ft.openai_ft import job_request

    params = inspect.signature(openai.OpenAI(api_key="x").fine_tuning.jobs.create).parameters
    req = job_request("gpt-4.1-mini-2025-04-14", 3, "file-a", "file-b")
    assert set(req) <= set(params), set(req) - set(params)
    print(f"ok  openai fine_tuning.jobs.create({', '.join(req)})")


def main() -> None:
    for name in SCRIPTS:
        py_compile.compile(str(HERE / "ch19_ft" / f"{name}.py"), doraise=True)
    print(f"ok  py_compile {', '.join(SCRIPTS)}")
    for pkg in ("openai", "boto3", "sagemaker", "google-genai", "google-cloud-storage"):
        print(f"    {pkg}=={md.version(pkg)}")
    check_openai()
    check_boto()
    check_genai()
    check_sagemaker()


if __name__ == "__main__":
    main()
