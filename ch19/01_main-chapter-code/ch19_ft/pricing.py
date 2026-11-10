"""Managed fine-tuning prices and limits, researched on 2026-11-08, and the cost arithmetic.

Prices are USD per 1M tokens unless the field name says otherwise. Every row names its source;
re-check them before you spend money: two of the providers below changed their fine-tuning
offer between May and November 2026.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

ACCESSED = dt.date(2026, 11, 8)
HOURS_PER_MONTH = 730


@dataclass(frozen=True)
class FTOffer:
    """One way to fine-tune and then serve a model."""

    provider: str
    model: str
    training_per_m: float              # $ per 1M training tokens (dataset tokens x epochs)
    default_epochs: int
    input_per_m: float | None = None   # serving, per token (None: you rent a machine instead)
    output_per_m: float | None = None
    monthly_fixed: float = 0.0         # storage or an always-on endpoint, $ per month
    min_job_usd: float = 0.0
    shutdown: dt.date | None = None    # base model (and every model tuned from it) stops serving
    note: str = ""
    source: str = ""


OPENAI = "https://developers.openai.com/api/docs/pricing"
OPENAI_DEPRECATIONS = "https://developers.openai.com/api/docs/deprecations"
BEDROCK = "https://aws.amazon.com/bedrock/pricing/"
VERTEX = "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing"
SAGEMAKER = "https://aws.amazon.com/sagemaker/ai/pricing/"
TOGETHER = "https://docs.together.ai/docs/fine-tuning/pricing"
FIREWORKS = "https://fireworks.ai/pricing"
OPENAI_SHUTDOWN = dt.date(2027, 1, 14)

OFFERS: list[FTOffer] = [
    FTOffer("OpenAI", "gpt-4.1-mini-2025-04-14", 5.00, 3, 0.80, 3.20,
            shutdown=OPENAI_SHUTDOWN,
            note="no new jobs after 2027-01-06; only orgs with recent fine-tuned inference",
            source=OPENAI),
    FTOffer("OpenAI", "gpt-4o-mini-2024-07-18", 3.00, 3, 0.30, 1.20, shutdown=OPENAI_SHUTDOWN,
            note="same wind-down", source=OPENAI),
    FTOffer("OpenAI", "gpt-4.1-2025-04-14", 25.00, 3, 3.00, 12.00, shutdown=OPENAI_SHUTDOWN,
            note="same wind-down", source=OPENAI),
    FTOffer("Bedrock", "Nova Micro", 1.00, 2, 0.035, 0.14, monthly_fixed=1.95,
            note="on-demand custom model deployment, us-east-1; storage $1.95/month",
            source=BEDROCK),
    FTOffer("Bedrock", "Nova Lite", 2.00, 2, 0.06, 0.24, monthly_fixed=1.95,
            note="on-demand, us-east-1", source=BEDROCK),
    FTOffer("Bedrock", "Nova 2 Lite", 3.78, 2, 0.30, 2.50, monthly_fixed=1.95,
            note="on-demand, us-east-1", source=BEDROCK),
    FTOffer("Bedrock", "Nova Pro", 8.00, 2, 0.80, 3.20, monthly_fixed=1.95,
            note="on-demand, us-east-1", source=BEDROCK),
    FTOffer("Vertex", "gemini-3.1-flash-lite", 3.00, 3, 0.25, 1.50,
            note="tuned = base price on the global endpoint, 1.5x on us/eu", source=VERTEX),
    FTOffer("Vertex", "gemini-3.5-flash-lite", 3.00, 3, 0.30, 2.50,
            note="tuned = base price on the global endpoint, 1.5x on us/eu", source=VERTEX),
    FTOffer("Vertex", "gemini-3.5-flash", 10.00, 3, 1.50, 9.00,
            note="tuned = base price on the global endpoint, 1.5x on us/eu", source=VERTEX),
    FTOffer("SageMaker", "Qwen3 8B LoRA (serverless SFT)", 0.50, 3,
            monthly_fixed=1.515 * HOURS_PER_MONTH,
            note="serve on an ml.g5.2xlarge endpoint, $1.515/h, always on", source=SAGEMAKER),
    FTOffer("Together", "Qwen3.5 9B LoRA", 0.34, 3, monthly_fixed=5.49 * HOURS_PER_MONTH,
            min_job_usd=4.00,
            note="$4 minimum per job; serverless LoRA discontinued: dedicated H100 $5.49/h",
            source=TOGETHER),
    FTOffer("Fireworks", "<=16B model LoRA", 0.50, 1, monthly_fixed=8.00 * HOURS_PER_MONTH,
            note="dedicated H100 $8/h, scales to zero after 1 h idle", source=FIREWORKS),
]

# The OpenAI models openai_ft.py accepts, by exact name.
OPENAI_FT: dict[str, FTOffer] = {o.model: o for o in OFFERS if o.provider == "OpenAI"}


def training_cost(tokens_per_epoch: int, epochs: int, price_per_m: float,
                  minimum: float = 0.0) -> float:
    """Billed training tokens are dataset tokens x epochs (OpenAI, Bedrock, Vertex all say so)."""
    if tokens_per_epoch < 0 or epochs < 1:
        raise ValueError("tokens must be >= 0 and epochs >= 1")
    return max(minimum, tokens_per_epoch * epochs * price_per_m / 1e6)


def cost_per_1k(prompt_tokens: float, completion_tokens: float, input_per_m: float,
                output_per_m: float) -> float:
    """$ per 1,000 requests from the average tokens per request."""
    return (prompt_tokens * input_per_m + completion_tokens * output_per_m) / 1e3


def monthly_cost(requests_per_month: float, per_1k: float, fixed: float = 0.0) -> float:
    return requests_per_month / 1e3 * per_1k + fixed


def offer_monthly(offer: FTOffer, requests_per_month: float, prompt_tokens: float,
                  completion_tokens: float) -> float:
    """Serving cost per month. Dedicated-GPU offers cost the same at any volume one GPU absorbs."""
    per_1k = 0.0
    if offer.input_per_m is not None:
        per_1k = cost_per_1k(prompt_tokens, completion_tokens, offer.input_per_m,
                             offer.output_per_m or 0.0)
    return monthly_cost(requests_per_month, per_1k, offer.monthly_fixed)


def breakeven(per_1k_a: float, fixed_a: float, per_1k_b: float, fixed_b: float) -> float | None:
    """Requests per month above which option B (higher fixed, lower per-request) is cheaper."""
    saving = per_1k_a - per_1k_b
    if saving <= 0:
        return None
    return max(0.0, (fixed_b - fixed_a) / saving * 1e3)
