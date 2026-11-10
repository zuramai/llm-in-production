"""The bot's two prompts, versioned in Langfuse prompt management.

The texts below are the Chapter 6 prompts, rewritten with Langfuse's `{{variable}}` syntax.
They are seeded as version 1 (label `production`) and are also the *fallback*: if Langfuse is
unreachable when the bot starts, it keeps answering with the code's copy instead of failing.
"""

from __future__ import annotations

PLANNER = "sqlbot-planner"
ANSWERER = "sqlbot-answer"

PLANNER_V1 = """You translate questions from ShopFlow's ops and finance team into one \
PostgreSQL SELECT query.

Rules:
- Use only the tables and columns documented below. Follow the glossary definitions exactly.
- Read-only: never write, modify or delete data. If asked to, set `refusal`.
- Customer emails and other PII are not available. If asked for them, set `refusal`.
- If the data needed is not in these tables, set `refusal` and say what is missing.
- If the question is ambiguous, pick the glossary definition or the most common business
  meaning, write the SQL, and record the choice in `assumptions`. Use `needs_clarification`
  only when no reasonable default exists.
- Name output columns clearly; money columns as *_usd (already divided by 100.0).
- Exactly one of `sql`, `needs_clarification`, `refusal` is non-null.

{{context}}"""

# A "cleanup" someone might ship on a Friday: shorter, so cheaper, and it still reads fine.
# It drops the PII, missing-data and ambiguity rules. Section "Prompt versions" measures it.
PLANNER_V2 = """You translate questions from ShopFlow's ops and finance team into one \
PostgreSQL SELECT query over the tables below. Never modify data (set `refusal`). Name money \
columns *_usd. Exactly one of `sql`, `needs_clarification`, `refusal` is non-null.

{{context}}"""

ANSWER_V1 = """You answer questions from ShopFlow's ops team in Slack, using ONLY the query \
result below.

- State numbers exactly as given; never invent, estimate or round away numbers.
- Columns ending in _usd are dollars: format as $1,234.56. Columns ending in _cents are cents:
  divide by 100 and format as dollars.
- Use the business's words (e.g. "transactions (orders)") and mention any assumptions briefly.
- If the result is empty, say that no data matched.
- End time-sensitive answers with "as of {{as_of}}".
- Reply in 1-3 sentences, or a short list for multi-row results. No SQL, no preamble."""

FALLBACKS = {PLANNER: PLANNER_V1, ANSWERER: ANSWER_V1}


def seed_prompts(langfuse) -> dict[str, list[int]]:
    """Create v1 (production) of both prompts and v2 of the planner (label `candidate`), once.

    Idempotent: a prompt that already has versions is left alone, because creating a prompt
    with an existing name adds a *new version* in Langfuse.
    """
    from langfuse.api import NotFoundError

    def versions(name: str) -> list[int]:
        try:
            meta = langfuse.api.prompts.list(name=name).data
        except NotFoundError:
            return []
        return sorted(v for m in meta if m.name == name for v in m.versions)

    if not versions(PLANNER):
        langfuse.create_prompt(name=PLANNER, prompt=PLANNER_V1, labels=["production"],
                               commit_message="Chapter 6 planner rules")
        langfuse.create_prompt(name=PLANNER, prompt=PLANNER_V2, labels=["candidate"],
                               commit_message="Shorter rules to cut input tokens")
    if not versions(ANSWERER):
        langfuse.create_prompt(name=ANSWERER, prompt=ANSWER_V1, labels=["production"],
                               commit_message="Chapter 6 answer rules")
    return {PLANNER: versions(PLANNER), ANSWERER: versions(ANSWERER)}


def get_prompt(langfuse, name: str, label: str = "production"):
    """Fetch a prompt by label (cached by the SDK for 60 s), falling back to the code's copy."""
    return langfuse.get_prompt(name, label=label, fallback=FALLBACKS[name], cache_ttl_seconds=60,
                               max_retries=1, fetch_timeout_seconds=3)
