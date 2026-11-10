"""Chapter 19: managed fine-tuning of the ShopFlow triage model (OpenAI, Bedrock, Vertex, SageMaker).

* ``convert``: Chapter 18's split -> OpenAI, Bedrock and Vertex JSONL, plus validators (executed).
* ``pricing``: the dated price table and the cost / break-even arithmetic (executed).
* ``evaluate``: the same 210-ticket eval as Chapter 18, against any OpenAI-compatible endpoint.
* ``openai_ft``: upload, launch, poll and evaluate an OpenAI fine-tuning job (needs OPENAI_API_KEY).
* ``bedrock_ft``, ``vertex_ft``, ``sagemaker_ft``: cloud launchers (need cloud accounts).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent  # ch19/01_main-chapter-code
DATA = HERE / "data"
FORMATS_DIR = HERE / "formats"
RESULTS = HERE / "results"


def previous() -> Any:
    """Code from Chapters 2 and 18, loaded under a name no other chapter uses."""
    name = "ch19_previous_chapters"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "previous_chapters.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
