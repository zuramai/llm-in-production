"""Data shapes shared by the reviewer, the eval and the GitHub poster."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Severity = Literal["low", "medium", "high", "critical"]
Category = Literal[
    "security", "correctness", "performance", "concurrency", "maintainability", "style", "tests"
]
SEVERITY_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class ReviewComment(BaseModel):
    """One finding, in the shape the model must return and GitHub needs."""

    model_config = ConfigDict(extra="forbid")

    path: str
    line: int
    severity: Severity
    category: Category
    body: str
    confidence: float  # 0..1, self-reported; we clamp it in code (strict mode has no min/max)


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comments: list[ReviewComment]


def strict_schema(model: type[BaseModel]) -> dict:
    """JSON schema in the shape OpenAI-style strict mode wants.

    Every object needs ``additionalProperties: false`` and every property listed as required,
    including objects nested under ``$defs``.
    """
    schema = model.model_json_schema()

    def fix(node):
        if isinstance(node, dict):
            if "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    return schema


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "ReviewResult", "strict": True, "schema": strict_schema(ReviewResult)},
}
