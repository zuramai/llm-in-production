import json
import sys
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest
from fastapi.testclient import TestClient

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parents[1]
sys.path.insert(0, str(CODE / "services" / "chat"))

from ch20_chat.app import create_app
from ch20_chat.canary import choose_version
from ch20_chat.core import FAQ_PATH, load_prompts
from ch20_chat.sessions import MemorySessionStore
from ch20_chat.settings import load_settings


class FakeUsage(SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return dict(vars(self))


class FakeCompletions:
    """Records every request; streams 'reply-N' as two chunks plus a usage chunk."""

    def __init__(self, fail_with=None):
        self.calls: list[dict] = []
        self.fail_with = fail_with

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_with:
            raise self.fail_with
        n = len(self.calls)

        def chunk(content=None, usage=None):
            choices = [] if content is None else [
                SimpleNamespace(delta=SimpleNamespace(content=content))]
            return SimpleNamespace(choices=choices, usage=usage)

        return iter([chunk("reply-"), chunk(str(n)),
                     chunk(usage=FakeUsage(prompt_tokens=10, completion_tokens=2))])


def make(env=None, store=None, fail_with=None):
    completions = FakeCompletions(fail_with)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    settings = load_settings({"LLM_BASE_URL": "http://x/v1", "LLM_API_KEY": "k",
                              "LLM_MODEL": "m", **(env or {})})
    app = create_app(settings, client=client, store=store or MemorySessionStore())
    return TestClient(app), completions


def events(resp):
    return [json.loads(line[5:]) for line in resp.text.splitlines() if line.startswith("data:")]


def test_faq_copy_matches_the_dataset():
    # The service ships its own copy of the FAQ; it must not drift from the source of truth.
    assert FAQ_PATH.read_text() == (REPO / "datasets/shopflow/docs/faq.md").read_text()


def test_every_prompt_version_embeds_the_faq():
    prompts = load_prompts()
    assert {"v1", "v2"} <= set(prompts)
    for text in prompts.values():
        assert "{faq}" not in text
        assert "## Returns and refunds" in text


def test_conversation_is_remembered_and_system_prompt_is_not_stored():
    store = MemorySessionStore()
    client, completions = make(store=store)
    client.post("/chat", json={"session_id": "s1", "message": "first"})
    r = client.post("/chat", json={"session_id": "s1", "message": "second"})
    assert events(r)[-1]["done"] is True
    sent = completions.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[2]["content"] == "reply-1"
    assert [m["role"] for m in store.get("s1")] == ["user", "assistant", "user", "assistant"]


def test_prompt_version_follows_the_router_and_is_reported():
    env = {"PROMPT_CANARY": "v2", "CANARY_PERCENT": "50"}
    client, completions = make(env)
    prompts = load_prompts()
    for sid in [f"user-{i}" for i in range(12)]:
        r = client.post("/chat", json={"session_id": sid, "message": "hi"})
        expected = choose_version(sid, "v1", "v2", 50)
        assert r.headers["x-prompt-version"] == expected
        assert events(r)[-1]["prompt_version"] == expected
        assert completions.calls[-1]["messages"][0]["content"] == prompts[expected]


def test_unknown_prompt_version_fails_at_startup():
    with pytest.raises(ValueError, match="unknown prompt version 'v9'"):
        make({"PROMPT_STABLE": "v9"})


def test_readiness_reflects_the_session_store_but_liveness_does_not():
    class DownStore(MemorySessionStore):
        async def aping(self):
            raise OSError("db down")

    client, _ = make(store=DownStore())
    assert client.get("/readyz").status_code == 503
    assert client.get("/healthz").status_code == 200


def test_provider_error_becomes_an_error_event_and_nothing_is_stored():
    store = MemorySessionStore()
    err = openai.APIConnectionError(request=None)
    client, _ = make(store=store, fail_with=err)
    r = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert r.status_code == 200
    assert "error" in events(r)[-1]
    assert store.get("s1") == []


@pytest.mark.parametrize("message", ["", "x" * 4001])
def test_input_limits(message):
    client, completions = make()
    assert client.post("/chat", json={"session_id": "s", "message": message}).status_code == 422
    assert completions.calls == []


def test_reasoning_effort_and_max_tokens_are_passed_through():
    client, completions = make({"REASONING_EFFORT": "none", "MAX_OUTPUT_TOKENS": "300"})
    client.post("/chat", json={"session_id": "s", "message": "hi"})
    assert completions.calls[0]["reasoning_effort"] == "none"
    assert completions.calls[0]["max_tokens"] == 300


def test_probes_run_on_the_event_loop_not_in_the_thread_pool():
    # Sync handlers share the 40-thread pool with the /chat streams; under load a sync probe
    # times out and liveness restarts a healthy pod. Probes must be `async def`.
    import inspect

    client, _ = make()
    endpoints = {r.path: r.endpoint for r in client.app.routes if hasattr(r, "endpoint")}
    assert inspect.iscoroutinefunction(endpoints["/healthz"])
    assert inspect.iscoroutinefunction(endpoints["/readyz"])
