import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch14_sqlbot import prompts


def _vars(template: str) -> set[str]:
    return set(re.findall(r"{{\s*(\w+)\s*}}", template))


def test_templates_use_langfuse_variables_only():
    assert _vars(prompts.PLANNER_V1) == {"context"}
    assert _vars(prompts.PLANNER_V2) == {"context"}
    assert _vars(prompts.ANSWER_V1) == {"as_of"}
    # A leftover str.format placeholder from Chapter 6 would reach the model verbatim.
    for t in (prompts.PLANNER_V1, prompts.PLANNER_V2, prompts.ANSWER_V1):
        assert not re.search(r"(?<!{){\w+}(?!})", t)


def test_candidate_planner_is_shorter_and_drops_the_pii_rule():
    assert len(prompts.PLANNER_V2) < len(prompts.PLANNER_V1) / 2
    assert "PII" in prompts.PLANNER_V1 and "PII" not in prompts.PLANNER_V2


def test_get_prompt_falls_back_to_code_copy_when_langfuse_is_down():
    from langfuse import Langfuse

    # Port 9 (discard) refuses connections: Langfuse is "down".
    lf = Langfuse(base_url="http://127.0.0.1:9", public_key="pk-x", secret_key="sk-x",
                  tracing_enabled=False)
    planner = prompts.get_prompt(lf, prompts.PLANNER)
    assert planner.is_fallback
    compiled = planner.compile(context="## Tables\n- orders")
    assert compiled.endswith("## Tables\n- orders")
    assert "{{" not in compiled
    answerer = prompts.get_prompt(lf, prompts.ANSWERER)
    assert "as of 9:00 AM EST" in answerer.compile(as_of="9:00 AM EST")
