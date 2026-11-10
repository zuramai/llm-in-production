"""Pipeline control flow with a scripted fake client: no network, no model."""

import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch08_eval import load_run, save_run
from ch08_extract import Extraction
from ch08_invoice import Invoice, LineItem
from ch08_pipeline import ModelSpec, process_file

TODAY = date(2026, 11, 7)


def good() -> Invoice:
    return Invoice(supplier="Kettleby Homeware Ltd", invoice_no="K-12345", date="2026-01-10",
                   currency="GBP",
                   line_items=[LineItem(sku="KH-1001", desc="Mug", qty=12, unit_price_cents=480)],
                   subtotal_cents=5760, tax_cents=1152, total_cents=6912)


class FakeClient:
    """Answers each chat.completions.create call with the next scripted reply."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        content = self.replies.pop(0)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50,
                                model_extra={"cost": 0.001})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                     finish_reason="stop")], usage=usage)


def scan(tmp_path: Path) -> Path:
    from PIL import Image
    p = tmp_path / "scan.png"
    Image.new("L", (40, 40), 255).save(p)
    return p


def test_valid_first_answer_is_accepted_with_one_call(tmp_path):
    fake = FakeClient([good().model_dump_json()])
    out = process_file(scan(tmp_path), text=ModelSpec(fake, "m"),
                       strong=ModelSpec(fake, "big"), today=TODAY)
    assert out.status == "accepted" and len(fake.calls) == 1
    assert out.invoice == good() and out.cost_usd == 0.001


def test_repair_conversation_contains_the_broken_rule_and_the_previous_answer(tmp_path):
    bad = good().model_copy(update={"total_cents": 6913})  # one cent off
    fake = FakeClient([bad.model_dump_json(), good().model_dump_json()])
    out = process_file(scan(tmp_path), text=ModelSpec(fake, "m"),
                       strong=ModelSpec(fake, "big"), today=TODAY)
    assert out.status == "accepted"
    assert [a.stage for a in out.attempts] == ["first", "repair"]
    repair_msgs = fake.calls[1]["messages"]
    assert repair_msgs[-2] == {"role": "assistant", "content": bad.model_dump_json()}
    assert "total_sum" in repair_msgs[-1]["content"]
    assert fake.calls[1]["model"] == "m"


def test_escalation_uses_the_strong_client_and_its_own_kwargs(tmp_path):
    bad = good().model_copy(update={"subtotal_cents": 5761, "total_cents": 6913})
    local = FakeClient([bad.model_dump_json()] * 2)
    hosted = FakeClient([good().model_dump_json()])
    out = process_file(scan(tmp_path), text=ModelSpec(local, "qwen3:8b"),
                       vision=ModelSpec(local, "gemma3:4b", {"temperature": 0}, max_side=20),
                       strong=ModelSpec(hosted, "luna"), today=TODAY)
    assert out.status == "accepted" and [a.stage for a in out.attempts][-1] == "escalate"
    assert [c["model"] for c in local.calls] == ["gemma3:4b", "gemma3:4b"]
    assert local.calls[0]["temperature"] == 0 and "temperature" not in hosted.calls[0]
    # by default the strong model reads the page from scratch: no weak answer to anchor on
    assert [m["role"] for m in hosted.calls[0]["messages"]] == ["system", "user"]


def test_escalation_with_context_shows_the_weak_answer_and_broken_rules(tmp_path):
    bad = good().model_copy(update={"subtotal_cents": 5761, "total_cents": 6913})
    local = FakeClient([bad.model_dump_json()] * 2)
    hosted = FakeClient([good().model_dump_json()])
    process_file(scan(tmp_path), text=ModelSpec(local, "m"), strong=ModelSpec(hosted, "luna"),
                 escalate_fresh=False, today=TODAY)
    msgs = hosted.calls[0]["messages"]
    assert msgs[-2] == {"role": "assistant", "content": bad.model_dump_json()}
    assert "line_sum" in msgs[-1]["content"]


def test_still_broken_after_escalation_goes_to_review_with_reasons(tmp_path):
    bad = good().model_copy(update={"subtotal_cents": 5761, "total_cents": 6913})
    fake = FakeClient([bad.model_dump_json()] * 3)
    out = process_file(scan(tmp_path), text=ModelSpec(fake, "m"),
                       strong=ModelSpec(fake, "big"), today=TODAY)
    assert out.status == "review"
    assert [c["model"] for c in fake.calls] == ["m", "m", "big"]
    assert {v.rule for v in out.reasons} == {"line_sum"}
    assert out.cost_usd == 0.003


def test_unparseable_answer_is_a_violation_not_a_crash(tmp_path):
    fake = FakeClient(["Sorry, I can't read this.", "still no", "nope"])
    out = process_file(scan(tmp_path), text=ModelSpec(fake, "m"), today=TODAY)
    assert out.status == "review" and out.invoice is None
    assert [v.rule for v in out.reasons] == ["parse_error"]
    assert len(fake.calls) == 2  # first + repair, no strong model configured


def test_run_roundtrips_through_json(tmp_path):
    ex = Extraction(method="vision", model="m", invoice=good(), raw="{}", latency_s=1.5,
                    prompt_tokens=10, completion_tokens=5, cost_usd=0.0002, notes={"k": 1})
    failed = Extraction(method="text", model="m", error="parse_error: x")
    path = tmp_path / "r.json"
    save_run(path, {"config": "c"}, {"b.png": ex, "a.pdf": failed})
    meta, back = load_run(path)
    assert meta == {"config": "c"}
    assert back["b.png"] == ex and back["a.pdf"] == failed
    assert list(json.loads(path.read_text())["files"]) == ["a.pdf", "b.png"]  # stable order
