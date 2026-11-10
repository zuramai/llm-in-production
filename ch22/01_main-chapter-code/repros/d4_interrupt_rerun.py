"""D4: LangGraph resumes an interrupted node from its first line: side effects run twice."""

from __future__ import annotations

from typing import TypedDict

from ch22_common import Result, timed
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

ID, TITLE, LAYER, KIND = "D4", "interrupt() re-runs the node on resume", "Agents & tools", "local"


class State(TypedDict, total=False):
    order_id: int
    approved: bool


def build(side_effects: list[str], fixed: bool):
    def notify_then_ask(state: State) -> State:
        # Naive: the side effect sits above interrupt(), so it runs again on resume.
        side_effects.append(f"emailed customer about refund for {state['order_id']}")
        answer = interrupt({"question": f"Approve refund for order {state['order_id']}?"})
        return {"approved": answer == "yes"}

    def ask(state: State) -> State:
        answer = interrupt({"question": f"Approve refund for order {state['order_id']}?"})
        return {"approved": answer == "yes"}

    def notify(state: State) -> State:
        # Fixed: the side effect lives in its own node, after the approval checkpoint.
        side_effects.append(f"emailed customer about refund for {state['order_id']}")
        return {}

    g = StateGraph(State)
    if fixed:
        g.add_node("ask", ask)
        g.add_node("notify", notify)
        g.add_edge(START, "ask")
        g.add_edge("ask", "notify")
        g.add_edge("notify", END)
    else:
        g.add_node("notify_then_ask", notify_then_ask)
        g.add_edge(START, "notify_then_ask")
        g.add_edge("notify_then_ask", END)
    return g.compile(checkpointer=InMemorySaver())


def run_once(fixed: bool) -> tuple[int, bool]:
    effects: list[str] = []
    graph = build(effects, fixed)
    cfg = {"configurable": {"thread_id": f"refund-{fixed}"}}
    graph.invoke({"order_id": 4512}, cfg)          # pauses at interrupt()
    final = graph.invoke(Command(resume="yes"), cfg)  # human said yes
    return len(effects), final["approved"]


@timed
def run() -> Result:
    naive_effects, naive_ok = run_once(fixed=False)
    fixed_effects, fixed_ok = run_once(fixed=True)
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=naive_effects == 2,
        fixed=fixed_effects == 1 and fixed_ok,
        evidence=f"one approval: side effect above interrupt() ran {naive_effects}x "
                 f"(approved={naive_ok}); side effect in its own node after approval ran "
                 f"{fixed_effects}x",
    )
