"""Talk to the refund agent from the terminal. Each command is a separate process: the only
thing that survives between them is the Postgres checkpoint (schema `ch10`).

    uv run python ch10_cli.py ask t-42 "Order 123 arrived damaged, refund $25"
    uv run python ch10_cli.py pending
    uv run python ch10_cli.py decide t-42 --approve --role agent
    uv run python ch10_cli.py decide t-42 --reject --role team_lead --note "no photo"
"""

from __future__ import annotations

import argparse
import json
import os

import ch10_setup
from ch10_graph import AgentConfig, Turn, build_graph, run

from llmprod import connect


def _graph(saver, model: str):
    return build_graph(AgentConfig(model=model), saver)


def show(turn: Turn) -> None:
    if turn.pending:
        print("PAUSED, waiting for a human:")
        print(json.dumps(turn.pending[0], indent=2))
    else:
        print(turn.answer)


def pending_threads(graph) -> list[tuple[str, dict]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT thread_id, max(checkpoint_id) AS last FROM ch10.checkpoints "
            "WHERE checkpoint_ns = '' GROUP BY thread_id ORDER BY last DESC LIMIT 200"
        ).fetchall()
    out = []
    for (thread_id, _last) in rows:
        snap = graph.get_state({"configurable": {"thread_id": thread_id}})
        if snap.interrupts:
            out.append((thread_id, snap.interrupts[0].value))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=os.environ.get("CH10_MODEL", "openai/gpt-6-luna"))
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("thread")
    a.add_argument("text")
    sub.add_parser("pending")
    d = sub.add_parser("decide")
    d.add_argument("thread")
    g = d.add_mutually_exclusive_group(required=True)
    g.add_argument("--approve", action="store_true")
    g.add_argument("--reject", action="store_true")
    d.add_argument("--role", required=True, choices=["agent", "team_lead", "finance"])
    d.add_argument("--note", default="")
    args = p.parse_args()

    ch10_setup.setup()
    with ch10_setup.checkpointer() as saver:
        graph = _graph(saver, args.model)
        if args.cmd == "ask":
            show(run(graph, args.thread, user=args.text))
        elif args.cmd == "pending":
            for thread_id, req in pending_threads(graph):
                print(f"{thread_id}: {req['amount_usd']} on order {req['order_id']} "
                      f"({req['reason']}), needs {req['required_approver']}")  # fmt: skip
        else:
            decision = {"approved": args.approve, "role": args.role, "note": args.note}
            show(run(graph, args.thread, decision=decision))


if __name__ == "__main__":
    main()
