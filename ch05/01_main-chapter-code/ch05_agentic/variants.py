"""The three systems we compare, behind one interface: `run(question) -> Result`.

    ch04_rag    Chapter 4's final pipeline: retrieve 10 chunks for the question, answer from them.
    tool_loop   A plain function-calling loop: the model calls search_docs / run_sql until it
                answers (at most MAX_TURNS model calls).
    graph       The LangGraph agent in graph.py.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field

from .graph import RECURSION_LIMIT, build_graph, initial_state
from .llm import LLM, Usage
from .prev import prev
from .prompts import ANSWER
from .tools import (
    SCHEMA,
    DocEvidence,
    DocSearch,
    SqlEvidence,
    check_citations,
    now_text,
    run_sql,
)

MAX_TURNS = 8


@dataclass
class Result:
    variant: str
    model: str
    question: str
    answer: str
    latency_ms: int
    usage: dict
    citations: dict
    steps: list[dict] = field(default_factory=list)  # what ran, in order
    evidence: list[str] = field(default_factory=list)  # citation tags of everything retrieved
    sql_errors: int = 0  # queries the guard or Postgres rejected
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


# -------------------------------------------------------------------------------- ch04 RAG
def ch04_rag(question: str, llm: LLM, docs: DocSearch, today: str | None = None) -> Result:
    """Chapter 4 unchanged: the question goes straight to retrieval, no tools, no database."""
    t0 = time.perf_counter()
    usage = Usage()
    hits = docs.search(question, k=prev.RAG_K)
    context = "\n\n---\n\n".join(h.render() for h in hits)
    text = llm.text([{"role": "system", "content": prev.RAG_SYSTEM},
                     {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"}],
                    usage, "answer")
    return Result("ch04_rag", llm.model, question, text,
                  round((time.perf_counter() - t0) * 1000), usage.as_dict(),
                  check_citations(text, hits), [{"step": "retrieve", "query": question}],
                  [h.citation for h in hits])


# ------------------------------------------------------------------------------- tool loop
TOOLS = [
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search ShopFlow's policies, FAQ and internal handbook. Returns chunks "
                       "tagged [doc:path#N].",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["query"],
                       "properties": {"query": {"type": "string",
                                                "description": "a short search query"}}}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run ONE read-only PostgreSQL SELECT on ShopFlow's database (schema in "
                       "the system prompt). Returns rows tagged [sql:N], or the error.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["sql"],
                       "properties": {"sql": {"type": "string"}}}}},
]

LOOP_SYSTEM = """{answer_rules}

Gather evidence with the tools before answering: a question about a specific order, customer \
or ticket usually needs the facts from run_sql AND the rule from search_docs.

{schema}"""


def tool_loop(question: str, llm: LLM, docs: DocSearch, today: str | None = None,
              max_turns: int = MAX_TURNS) -> Result:
    """The hand-written agent loop most teams write first."""
    from llmprod import get_settings

    today = today or now_text()
    tz = get_settings().business_timezone
    t0 = time.perf_counter()
    usage = Usage()
    evidence: list[DocEvidence | SqlEvidence] = []
    steps: list[dict] = []
    messages: list[dict] = [
        {"role": "system", "content": LOOP_SYSTEM.format(
            answer_rules=ANSWER.format(today=today), schema=SCHEMA.format(tz=tz))},
        {"role": "user", "content": question},
    ]
    text, error = "", None
    for turn in range(max_turns):
        last = turn == max_turns - 1
        resp = llm.complete(messages, usage, f"turn{turn + 1}", tools=TOOLS,
                            tool_choice="none" if last else "auto")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            text = (msg.content or "").strip()
            break
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            out = _call_tool(call.function.name, call.function.arguments, docs, evidence)
            steps.append({"step": call.function.name, "args": call.function.arguments[:300]})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
    else:
        error = "max_turns"
    return Result("tool_loop", llm.model, question, text,
                  round((time.perf_counter() - t0) * 1000), usage.as_dict(),
                  check_citations(text, evidence), steps, [e.citation for e in evidence],
                  sum(isinstance(e, SqlEvidence) and e.error is not None for e in evidence),
                  error)


def _call_tool(name: str, arguments: str, docs: DocSearch,
               evidence: list[DocEvidence | SqlEvidence]) -> str:
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return f"ERROR: arguments are not valid JSON: {e}"
    if name == "search_docs":
        seen = {e.citation for e in evidence}
        hits = docs.search(str(args.get("query", "")))
        evidence += [h for h in hits if h.citation not in seen]
        return "\n\n---\n\n".join(h.render() for h in hits) or "no results"
    if name == "run_sql":
        n = 1 + sum(isinstance(e, SqlEvidence) for e in evidence)
        sql = str(args.get("sql", ""))
        try:
            ev = SqlEvidence(f"[sql:{n}]", "run_sql", sql, run_sql(sql))
        except (prev.UnsafeSQLError, prev.QueryError) as e:
            ev = SqlEvidence(f"[sql:{n}]", "run_sql", sql, [], str(e))
        evidence.append(ev)
        return ev.render()
    return f"ERROR: unknown tool {name}"


# ----------------------------------------------------------------------------------- graph
class GraphAgent:
    """Holds one compiled graph (compiling per question would waste ~ms and hide the reuse)."""

    def __init__(self, llm: LLM, docs: DocSearch, checkpointer=None):
        self.llm = llm
        self.app = build_graph(llm, docs, checkpointer=checkpointer)

    def __call__(self, question: str, today: str | None = None,
                 thread_id: str | None = None) -> Result:
        today = today or now_text()
        config = {"recursion_limit": RECURSION_LIMIT,
                  "configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
        t0 = time.perf_counter()
        error = None
        try:
            state = self.app.invoke(initial_state(question, today), config)
        except Exception as e:  # noqa: BLE001 - recorded per question, the eval goes on
            state, error = {}, f"{type(e).__name__}: {e}"
        usage = Usage()
        for c in state.get("llm_calls", []):
            usage.calls += 1
            usage.prompt_tokens += c["prompt_tokens"]
            usage.completion_tokens += c["completion_tokens"]
            usage.cost_usd += c["cost_usd"]
            usage.llm_ms += c["ms"]
        steps = [*state.get("trace", []), *[{"query": q} for q in state.get("queries", [])],
                 *[{"grade": g} for g in state.get("grades", [])]]
        evidence = state.get("evidence", [])
        return Result("graph", self.llm.model, question, state.get("answer", ""),
                      round((time.perf_counter() - t0) * 1000), usage.as_dict(),
                      state.get("citations", {}), steps, [e["citation"] for e in evidence],
                      sum(e["kind"] == "sql" and e.get("error") is not None for e in evidence),
                      error)


def make_runner(variant: str, llm: LLM, docs: DocSearch):
    if variant == "graph":
        return GraphAgent(llm, docs)
    if variant == "tool_loop":
        return lambda q, today=None: tool_loop(q, llm, docs, today)
    if variant == "ch04_rag":
        return lambda q, today=None: ch04_rag(q, llm, docs, today)
    raise ValueError(f"unknown variant {variant!r}")
