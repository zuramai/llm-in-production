"""Human approval queue for refund requests the assistant proposed.

    uv run python approve.py list [--status pending|approved|rejected|all]
    uv run python approve.py approve 3 --role team_lead [--note "ticket 4411"]
    uv run python approve.py reject 3 --role agent --note "outside return window"

Limits (handbook): agent <= $100.00, team_lead <= $500.00, finance above. `--role` stands in
for real authentication; in production the role comes from SSO, never from a flag.
"""

from __future__ import annotations

import argparse
import sys

import ch09_refunds as refunds
import ch09_setup


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no refund requests)")
        return
    print(f"{'id':>3}  {'order':>6}  {'amount':>10}  {'reason':<14} {'status':<9} {'needs':<9} key")
    for r in rows:
        print(
            f"{r['id']:>3}  {r['order_id']:>6}  {refunds.usd(r['amount_cents']):>10}  "
            f"{r['reason']:<14} {r['status']:<9} {r['required_approver']:<9} {r['idempotency_key']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list", help="show refund requests")
    p_list.add_argument(
        "--status", default="pending", choices=["pending", "approved", "rejected", "all"]
    )
    for name in ("approve", "reject"):
        p = sub.add_parser(name)
        p.add_argument("refund_id", type=int)
        p.add_argument("--role", required=True, choices=sorted(refunds.ROLE_RANK))
        p.add_argument("--note", default="")
    args = parser.parse_args(argv)

    ch09_setup.setup_tables()
    if args.cmd == "list":
        _print_table(refunds.list_requests(None if args.status == "all" else args.status))
        return 0
    try:
        row = refunds.decide(
            args.refund_id, args.role, approve=args.cmd == "approve", note=args.note
        )
    except refunds.RefundError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(
        f"request {row['id']} {row['status']} by {row['decided_by_role']}: "
        f"{refunds.usd(row['amount_cents'])} on order {row['order_id']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
