import sys
from pathlib import Path

import pytest

SERVICES = Path(__file__).resolve().parents[1] / "services"
sys.path.insert(0, str(SERVICES / "chat"))
sys.path.insert(0, str(SERVICES / "sqlbot"))

from ch20_chat.settings import load_settings as load_chat
from ch20_sqlbot.settings import load_settings as load_sqlbot

BASE = {"LLM_BASE_URL": "http://llm/v1", "LLM_API_KEY": "k", "LLM_MODEL": "m"}


def test_chat_defaults():
    s = load_chat(BASE)
    assert (s.prompt_stable, s.prompt_canary, s.canary_percent) == ("v1", None, 0.0)
    assert s.session_store == "memory"
    assert s.reasoning_effort is None


def test_chat_missing_required_names_every_missing_variable():
    with pytest.raises(ValueError, match="LLM_API_KEY, LLM_MODEL"):
        load_chat({"LLM_BASE_URL": "x", "LLM_API_KEY": "  "})


@pytest.mark.parametrize("percent", ["-1", "100.1", "250"])
def test_chat_rejects_out_of_range_percent(percent):
    with pytest.raises(ValueError, match="CANARY_PERCENT"):
        load_chat({**BASE, "PROMPT_CANARY": "v2", "CANARY_PERCENT": percent})


def test_chat_percent_without_canary_is_a_config_error():
    with pytest.raises(ValueError, match="needs PROMPT_CANARY"):
        load_chat({**BASE, "CANARY_PERCENT": "10"})


def test_chat_boundary_percents_are_valid():
    assert load_chat({**BASE, "PROMPT_CANARY": "v2", "CANARY_PERCENT": "100"}).canary_percent == 100
    assert load_chat({**BASE, "CANARY_PERCENT": "0"}).canary_percent == 0


def test_chat_postgres_store_needs_a_url():
    with pytest.raises(ValueError, match="SESSIONS_DATABASE_URL"):
        load_chat({**BASE, "SESSION_STORE": "postgres"})
    with pytest.raises(ValueError, match="SESSION_STORE"):
        load_chat({**BASE, "SESSION_STORE": "redis"})


def test_chat_empty_value_means_unset():
    # Kustomize overlays blank a key with `REASONING_EFFORT=`; that must mean "provider default".
    assert load_chat({**BASE, "REASONING_EFFORT": ""}).reasoning_effort is None


def test_sqlbot_needs_only_the_read_only_url():
    s = load_sqlbot({**BASE, "DATABASE_URL_RO": "postgresql://ro@db/shopflow"})
    assert s.database_url_ro.startswith("postgresql://ro@")
    assert not hasattr(s, "database_url")
    with pytest.raises(ValueError, match="DATABASE_URL_RO"):
        load_sqlbot(BASE)
