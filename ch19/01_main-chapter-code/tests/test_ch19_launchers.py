import datetime as dt
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch19_ft import bedrock_ft, convert, openai_ft, pricing

RECORD = {
    "id": 1, "subject": "Where is my order?", "body": "Order #12 is late.", "channel": "email",
    "category": "shipping", "priority": "medium", "order_id": 12, "hard": False,
    "sentiment": "neutral", "summary": "Order 12 is late.",
}


@pytest.fixture(autouse=True)
def offline_tokens(monkeypatch):
    monkeypatch.setattr(convert, "_encoding", lambda: SimpleNamespace(encode=str.split))


def write_openai(path, n, answer=None):
    rows = [convert.to_openai({**RECORD, "subject": f"Order {i} late"}) for i in range(n)]
    for r in rows:
        r["messages"][-1]["content"] = answer or r["messages"][-1]["content"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# --------------------------------------------------------------------------- pricing


def test_training_cost_is_tokens_times_epochs_with_a_minimum():
    assert pricing.training_cost(146_538, 3, 5.00) == pytest.approx(2.19807)
    assert pricing.training_cost(146_538, 3, 0.34, minimum=4.00) == 4.00  # Together's floor
    with pytest.raises(ValueError):
        pricing.training_cost(1000, 0, 5.00)


def test_cost_per_1k_and_breakeven():
    luna = pricing.cost_per_1k(253, 98, 0.10, 0.50)
    assert luna == pytest.approx(0.0743)
    # a tuned model at $0.0226 / 1k with $1.95 a month of storage pays off above ~34k requests
    assert pricing.breakeven(0.0794, 0.0, 0.0226, 1.95) == pytest.approx(34_331, rel=1e-3)
    assert pricing.breakeven(0.02, 0.0, 0.05, 1.95) is None  # never cheaper per request


def test_dedicated_endpoints_cost_the_same_at_any_volume():
    together = next(o for o in pricing.OFFERS if o.provider == "Together")
    low = pricing.offer_monthly(together, 1_000, 253, 98)
    high = pricing.offer_monthly(together, 1_000_000, 253, 98)
    assert low == high == pytest.approx(5.49 * 730)
    nova = next(o for o in pricing.OFFERS if o.model == "Nova Micro")
    assert pricing.offer_monthly(nova, 0, 253, 98) == pytest.approx(1.95)
    assert pricing.offer_monthly(nova, 1_000_000, 253, 98) > 20


def test_every_openai_offer_shares_the_wind_down_date():
    assert set(openai_ft.OPENAI_FT) == {"gpt-4.1-mini-2025-04-14", "gpt-4o-mini-2024-07-18",
                                        "gpt-4.1-2025-04-14"}
    assert {o.shutdown for o in openai_ft.OPENAI_FT.values()} == {dt.date(2027, 1, 14)}
    assert all(o.source.startswith("https://") for o in pricing.OFFERS)


# --------------------------------------------------------------------------- openai_ft


def test_plan_prices_the_job_and_counts_days_left(tmp_path):
    train = write_openai(tmp_path / "train.jsonl", 20)
    p = openai_ft.plan("gpt-4.1-mini-2025-04-14", 3, train, None, today=dt.date(2026, 11, 8))
    assert p["train_examples"] == 20 and p["val_examples"] == 0
    tokens = p["train_tokens_per_epoch"]
    assert p["estimated_training_usd"] == round(tokens * 3 * 5.00 / 1e6, 2)
    assert p["days_of_service_left"] == 67


def test_plan_refuses_a_bad_file_before_any_upload(tmp_path):
    train = write_openai(tmp_path / "train.jsonl", 20, answer='{"category": "billing"}')
    with pytest.raises(SystemExit, match="line 1: bad_answer"):
        openai_ft.plan("gpt-4.1-mini-2025-04-14", 3, train, None)


def test_plan_refuses_models_that_cannot_be_tuned(tmp_path):
    train = write_openai(tmp_path / "train.jsonl", 20)
    with pytest.raises(SystemExit, match="not fine-tunable"):
        openai_ft.plan("gpt-6-luna", 3, train, None)


def test_job_request_uses_the_method_field_and_omits_a_missing_val_file():
    req = openai_ft.job_request("gpt-4.1-mini-2025-04-14", 2, "file-t", None)
    assert req["method"] == {"type": "supervised",
                             "supervised": {"hyperparameters": {"n_epochs": 2}}}
    assert "validation_file" not in req and "hyperparameters" not in req
    assert len(req["suffix"]) <= 64


def test_new_events_prints_each_event_once_oldest_first():
    ev = lambda i, t: SimpleNamespace(id=f"ev{i}", created_at=t)
    seen: set[str] = set()
    first = openai_ft.new_events([ev(2, 20), ev(1, 10)], seen)
    assert [e.id for e in first] == ["ev1", "ev2"]
    again = openai_ft.new_events([ev(3, 30), ev(2, 20), ev(1, 10)], seen)
    assert [e.id for e in again] == ["ev3"]


def test_missing_key_explains_that_openrouter_cannot_fine_tune(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(SystemExit, match="OpenRouter"):
        openai_ft.openai_key()


# --------------------------------------------------------------------------- bedrock_ft


def test_nova_hyperparameters_are_strings_with_scaled_warmup():
    hp = bedrock_ft.hyperparameters(2, 972, "nova-micro")
    assert hp == {"epochCount": "2", "learningRate": "0.00001", "learningRateWarmupSteps": "2"}
    assert bedrock_ft.hyperparameters(2, 972, "nova-lite")["learningRateWarmupSteps"] == "6"
    assert bedrock_ft.hyperparameters(2, 500_000, "nova-lite")["learningRateWarmupSteps"] == "100"
    with pytest.raises(ValueError):
        bedrock_ft.hyperparameters(6, 972, "nova-micro")
