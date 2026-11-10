"""Agentic RAG as an explicit LangGraph state machine.

    route -> (retrieve | sql, in parallel) -> grade -> answer
                    ^                           |
                    +------ rewrite <-----------+   (insufficient evidence, at most MAX_LOOPS)

Every node is a plain function `state -> partial update`. The model never decides how many steps
run: the edges do. The state is plain JSON (dicts and lists), so any checkpointer can store it.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from . import prompts
from .llm import LLM, Usage
from .tools import (
    DocSearch,
    ask_sql,
    check_citations,
    evidence_dict,
    evidence_from_dict,
)

MAX_LOOPS = 2  # rewrite -> search again, at most twice
RECURSION_LIMIT = 25  # LangGraph's own backstop: supersteps per run (ours needs at most 10)


class AgentState(TypedDict, total=False):
    question: str
    today: str
    doc_query: str | None
    sql_question: str | None
    loops: int
    # Lists with a reducer: parallel nodes (retrieve and sql) append instead of overwriting.
    evidence: Annotated[list[dict], operator.add]
    queries: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]
    llm_calls: Annotated[list[dict], operator.add]
    grades: Annotated[list[dict], operator.add]  # every grade decision, to explain loops
    relevant: list[str]
    sufficient: bool
    missing: str
    answer: str
    citations: dict


class Route(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_query: str | None
    sql_question: str | None


class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant: list[str]
    sufficient: bool
    missing: str


def render(evidence: list[dict]) -> str:
    return "\n\n---\n\n".join(evidence_from_dict(e).render() for e in evidence) or "(none)"


def _timed(name: str, fn: Callable[[AgentState, Usage], dict]) -> Callable[[AgentState], dict]:
    """Wrap a node: measure it, and move its LLM usage into the state (no shared mutable
    counters, so parallel nodes can't race)."""

    def node(state: AgentState) -> dict:
        usage = Usage()
        t0 = time.perf_counter()
        update = fn(state, usage)
        ms = (time.perf_counter() - t0) * 1000
        return {**update, "trace": [{"node": name, "ms": round(ms), "llm_calls": usage.calls}],
                "llm_calls": usage.log}

    node.__name__ = name
    return node


def build_graph(llm: LLM, docs: DocSearch, checkpointer: Any = None, max_loops: int = MAX_LOOPS):
    """Compile the agent. `llm` and `docs` are closed over, so tests can pass fakes."""

    def route(state: AgentState, usage: Usage) -> dict:
        r = llm.structured(
            [{"role": "system", "content": prompts.ROUTE.format(today=state["today"])},
             {"role": "user", "content": state["question"]}], Route, usage, "route")
        return {"doc_query": r.doc_query, "sql_question": r.sql_question, "loops": 0,
                "queries": [{"step": "route", **r.model_dump()}]}

    def retrieve(state: AgentState, usage: Usage) -> dict:
        seen = {e["citation"] for e in state.get("evidence", [])}
        hits = [h for h in docs.search(state["doc_query"]) if h.citation not in seen]
        return {"evidence": [evidence_dict(h) for h in hits]}

    def sql(state: AgentState, usage: Usage) -> dict:
        n = 1 + sum(e["kind"] == "sql" for e in state.get("evidence", []))
        ev = ask_sql(state["sql_question"], llm, usage, f"[sql:{n}]", state["today"])
        return {"evidence": [evidence_dict(ev)]}

    def grade(state: AgentState, usage: Usage) -> dict:
        g = llm.structured(
            [{"role": "user", "content": prompts.GRADE.format(
                question=state["question"], evidence=render(state["evidence"]))}],
            Grade, usage, "grade")
        known = {e["citation"] for e in state["evidence"]}
        relevant = [t for t in g.relevant if t in known]
        return {"relevant": relevant, "sufficient": g.sufficient, "missing": g.missing,
                "grades": [{"sufficient": g.sufficient, "missing": g.missing,
                            "relevant": relevant}]}

    def rewrite(state: AgentState, usage: Usage) -> dict:
        tried = [{k: v for k, v in q.items() if k != "step"} for q in state["queries"]]
        facts = render([e for e in state["evidence"] if e["kind"] == "sql"])
        r = llm.structured(
            [{"role": "system", "content": prompts.REWRITE.format(
                missing=state["missing"], tried=tried, facts=facts, today=state["today"])},
             {"role": "user", "content": state["question"]}], Route, usage, "rewrite")
        return {"doc_query": r.doc_query, "sql_question": r.sql_question,
                "loops": state["loops"] + 1,
                "queries": [{"step": f"rewrite{state['loops'] + 1}", **r.model_dump()}]}

    def answer(state: AgentState, usage: Usage) -> dict:
        evidence = state.get("evidence", [])
        relevant = set(state.get("relevant") or [])
        used = [e for e in evidence if e["citation"] in relevant] or evidence
        text = llm.text(
            [{"role": "system", "content": prompts.ANSWER.format(today=state["today"])},
             {"role": "user", "content": f"Evidence:\n\n{render(used)}\n\n"
                                         f"Question: {state['question']}"}], usage, "answer")
        cites = check_citations(text, [evidence_from_dict(e) for e in evidence])
        return {"answer": text, "citations": cites}

    def dispatch(state: AgentState) -> list[str]:
        """After route/rewrite: fan out to the tools that have a query (both run in parallel,
        in the same superstep), or go straight to answer when neither is needed."""
        targets = [n for n, key in (("retrieve", "doc_query"), ("sql", "sql_question"))
                   if state.get(key)]
        return targets or ["answer"]

    def after_grade(state: AgentState) -> str:
        if state["sufficient"] or state["loops"] >= max_loops:
            return "answer"
        return "rewrite"

    g = StateGraph(AgentState)
    for name, fn in (("route", route), ("retrieve", retrieve), ("sql", sql), ("grade", grade),
                     ("rewrite", rewrite), ("answer", answer)):
        g.add_node(name, _timed(name, fn))
    g.add_edge(START, "route")
    g.add_conditional_edges("route", dispatch, ["retrieve", "sql", "answer"])
    g.add_conditional_edges("rewrite", dispatch, ["retrieve", "sql", "answer"])
    g.add_edge("retrieve", "grade")
    g.add_edge("sql", "grade")
    g.add_conditional_edges("grade", after_grade, ["answer", "rewrite"])
    g.add_edge("answer", END)
    return g.compile(checkpointer=checkpointer)


def initial_state(question: str, today: str) -> AgentState:
    return {"question": question, "today": today}
