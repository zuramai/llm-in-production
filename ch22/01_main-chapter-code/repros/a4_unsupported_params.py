"""A4: reasoning models reject (or a router silently drops) `temperature` and friends."""

from __future__ import annotations

import logging

import openai
from ch22_common import LUNA, SONNET, Result, chat, model_catalog, timed

ID, TITLE, LAYER, KIND = "A4", "Reasoning models reject temperature", "API & tokens", "hosted"
log = logging.getLogger(__name__)

STRICT = {"provider": {"require_parameters": True}}
SAMPLING = ("temperature", "top_p", "top_k", "seed", "frequency_penalty", "presence_penalty")


def fit_params(model_meta: dict, wanted: dict) -> tuple[dict, list[str]]:
    """Keep only parameters the model accepts; map reasoning_effort to a supported value.

    Returns (params to send, list of human-readable adjustments) so the caller can log them
    instead of discovering weeks later that `temperature=0` never did anything.
    """
    supported = set(model_meta.get("supported_parameters") or [])
    reasoning = model_meta.get("reasoning") or {}
    efforts = reasoning.get("supported_efforts") or []
    sent, notes = {}, []
    for key, value in wanted.items():
        if key == "reasoning_effort" and efforts and value not in efforts:
            # Mandatory-reasoning models have no "none": use the cheapest effort they allow.
            cheapest = efforts[-1]
            sent[key] = cheapest
            notes.append(f"reasoning_effort {value!r}->{cheapest!r}")
        elif key in SAMPLING and key not in supported:
            notes.append(f"dropped {key}")
        else:
            sent[key] = value
    return sent, notes


def _try(model: str, **kw) -> str:
    try:
        r = chat(model, [{"role": "user", "content": "Say OK."}], max_tokens=50, **kw)
        return f"200 {r.choices[0].message.content!r}"
    except openai.APIStatusError as e:
        return f"{e.status_code} {e.message[:70]}"


@timed
def run() -> Result:
    cat = model_catalog()
    obs = {
        "luna temperature=0.2": _try(LUNA, temperature=0.2),
        "luna temperature=0.2 + require_parameters": _try(LUNA, temperature=0.2, extra_body=STRICT),
        "sonnet reasoning_effort=none": _try(SONNET, reasoning_effort="none"),
    }
    fixed_obs = {}
    for model in (LUNA, SONNET):
        params, notes = fit_params(cat[model], {"temperature": 0.2, "reasoning_effort": "none"})
        log.warning("%s: %s", model, notes)
        fixed_obs[model] = (notes, _try(model, extra_body=STRICT, **params))
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=obs["luna temperature=0.2"].startswith("200")
        and not obs["luna temperature=0.2 + require_parameters"].startswith("200")
        and not obs["sonnet reasoning_effort=none"].startswith("200"),
        fixed=all(v[1].startswith("200") for v in fixed_obs.values()),
        evidence="; ".join(f"{k}: {v}" for k, v in obs.items())
        + " | fit_params: " + "; ".join(f"{k.split('/')[1]} {v[0]} -> {v[1][:10]}"
                                        for k, v in fixed_obs.items()),
        data={"naive": obs, "fixed": {k: list(v) for k, v in fixed_obs.items()}},
    )
