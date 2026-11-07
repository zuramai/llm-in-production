#!/usr/bin/env bash
# Build the sample repo the review bot is tested on.
#
#   ./make_sample_pr.sh [DEST_DIR] [GROUND_TRUTH_JSON]
#
# DEST_DIR defaults to ./sample_repo (gitignored). The script creates a fresh git repo with:
#   main                  a small, clean ShopFlow order service
#   chore/cleanup         main + 7 innocuous changes (refactors, docs, a test): a "clean" PR
#   feature/order-fixes   chore/cleanup + EXACTLY 8 planted bugs: the "buggy" PR
# Ground truth (bug id, file, line in the head version of the file) is written to
# GROUND_TRUTH_JSON (default: ./ground_truth.json). The bug lines are located with temporary
# markers that are stripped before the commit, so the code the model sees has no hints.
# Commit dates and authors are fixed, so the SHAs are reproducible.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/sample_repo}"
GT_OUT="${2:-$HERE/ground_truth.json}"

rm -rf "$DEST"
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
cd "$DEST"

export GIT_AUTHOR_NAME="ShopFlow Dev" GIT_AUTHOR_EMAIL="dev@shopflow.example"
export GIT_COMMITTER_NAME="ShopFlow Dev" GIT_COMMITTER_EMAIL="dev@shopflow.example"
git init -q -b main
git config commit.gpgsign false
commit() {  # commit "message" "ISO date"
  git add -A
  GIT_AUTHOR_DATE="$2" GIT_COMMITTER_DATE="$2" git commit -q -m "$1"
}

# ------------------------------------------------------------------ main: the clean service
mkdir -p service tests
touch service/__init__.py

cat > README.md <<'EOF'
# ShopFlow order service

Small internal service behind the support dashboard: order lookups, stock reservation and
payment capture.
EOF

cat > service/config.py <<'EOF'
"""Runtime settings for the ShopFlow order service."""

import os

BUSINESS_TZ = "America/New_York"
PAGE_SIZE = 20


def database_url() -> str:
    return os.environ["DATABASE_URL"]


STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "")
EOF

cat > service/db.py <<'EOF'
import psycopg

from .config import database_url


def connect() -> psycopg.Connection:
    return psycopg.connect(database_url(), autocommit=True)
EOF

cat > service/orders.py <<'EOF'
"""Order queries used by the support dashboard."""

from datetime import datetime
from zoneinfo import ZoneInfo

from .config import BUSINESS_TZ, PAGE_SIZE
from .db import connect


def get_order(order_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, customer_id, status, total_cents FROM orders WHERE id = %s",
            (order_id,),
        ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "customer_id": row[1], "status": row[2], "total_cents": row[3]}


def find_orders_by_status(status: str) -> list[tuple]:
    with connect() as conn:
        return conn.execute(
            "SELECT id, customer_id, total_cents FROM orders WHERE status = %s ORDER BY id",
            (status,),
        ).fetchall()


def page_of_orders(customer_id: int, page: int) -> list[tuple]:
    """Return one page of a customer's orders, newest first. `page` is 1-based."""
    offset = (page - 1) * PAGE_SIZE
    with connect() as conn:
        return conn.execute(
            "SELECT id, status, total_cents FROM orders WHERE customer_id = %s "
            "ORDER BY placed_at DESC LIMIT %s OFFSET %s",
            (customer_id, PAGE_SIZE, offset),
        ).fetchall()


def orders_placed_today() -> int:
    tz = ZoneInfo(BUSINESS_TZ)
    start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    with connect() as conn:
        row = conn.execute(
            "SELECT count(*) FROM orders WHERE placed_at >= %s", (start,)
        ).fetchone()
    return row[0]


def customer_totals(customer_ids: list[int]) -> dict[int, int]:
    """Lifetime spend in cents per customer, in one query."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT customer_id, sum(total_cents) FROM orders "
            "WHERE customer_id = ANY(%s) GROUP BY customer_id",
            (customer_ids,),
        ).fetchall()
    totals = {cid: 0 for cid in customer_ids}
    totals.update({cid: int(total) for cid, total in rows})
    return totals


def format_money(cents):
    return "$%.2f" % (cents / 100)
EOF

cat > service/inventory.py <<'EOF'
"""Stock lookups and reservation for checkout."""

from .db import connect


def available(product_id: int, warehouse: str = "main") -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT on_hand FROM inventory WHERE product_id = %s AND warehouse = %s",
            (product_id, warehouse),
        ).fetchone()
    return row[0] if row else 0


def reserve(product_id: int, qty: int, warehouse: str = "main") -> bool:
    """Atomically take `qty` units. Returns False when there is not enough stock."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE inventory SET on_hand = on_hand - %s "
            "WHERE product_id = %s AND warehouse = %s AND on_hand >= %s",
            (qty, product_id, warehouse, qty),
        )
        return cur.rowcount == 1


def low_stock(threshold: int = 5) -> list[int]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT product_id FROM inventory WHERE on_hand < %s ORDER BY 1",
            (threshold,),
        ).fetchall()
    return [r[0] for r in rows]
EOF

cat > service/gateway.py <<'EOF'
"""Payment provider client (stubbed in this sample)."""

import asyncio


async def charge(order_id: int, amount_cents: int) -> dict:
    await asyncio.sleep(0)  # stands in for the HTTP call to the provider
    return {"status": "succeeded", "order_id": order_id, "amount_cents": amount_cents}
EOF

cat > service/notify.py <<'EOF'
"""Customer notifications (stubbed in this sample)."""

import asyncio


async def send_receipt(order_id: int) -> None:
    await asyncio.sleep(0)  # stands in for the HTTP call to the email provider
EOF

cat > service/payments.py <<'EOF'
"""Capture payments and send receipts."""

from . import gateway, notify


async def capture(order_id: int, amount_cents: int) -> bool:
    result = await gateway.charge(order_id, amount_cents)
    if result["status"] != "succeeded":
        return False
    await notify.send_receipt(order_id)
    return True
EOF

commit "Add order service" "2026-10-20T10:00:00+00:00"

# -------------------------------------------------- the edit machinery (stdlib Python only)
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
run_edits() {  # run_edits FILE: exec FILE with edit()/write()/collect_markers() in scope
  PYTHONPATH="$TMP" python3 -c 'import sys, ch12_lib; exec(open(sys.argv[1]).read(), vars(ch12_lib))' "$1"
}

cat > "$TMP/ch12_lib.py" <<'PY'
import json
import pathlib
import os
import re

ROOT = pathlib.Path.cwd()


def edit(path, old, new):
    p = ROOT / path
    text = p.read_text()
    assert text.count(old) == 1, f"{path}: expected exactly one match for {old[:50]!r}"
    p.write_text(text.replace(old, new))


def write(path, text):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def collect_markers(out_path, extra):
    """Find `#@@bug|start|end:<id>` markers, strip them, and write the ground truth file.

    `bug` marks the primary line. Bugs that span several lines also carry `start` / `end`
    markers; a reviewer may point anywhere in [start, end] (plus the matching window).
    """
    found = {}
    for p in sorted(ROOT.glob("service/*.py")):
        lines = p.read_text().split("\n")
        for i, line in enumerate(lines, start=1):
            m = re.search(r"\s*#@@(bug|start|end):(\w+)\s*$", line)
            if m:
                kind, bug_id = m.groups()
                entry = found.setdefault(bug_id, {"id": bug_id, "file": str(p.relative_to(ROOT))})
                entry[{"bug": "line", "start": "start", "end": "end"}[kind]] = i
                lines[i - 1] = line[: m.start()]
        p.write_text("\n".join(lines))
    meta = extra["bugs"]
    bugs = []
    for bug_id, b in found.items():
        b.setdefault("start", b["line"])
        b.setdefault("end", b["line"])
        assert b["start"] <= b["line"] <= b["end"], b
        b.update(meta[bug_id])
        bugs.append(b)
    assert len(bugs) == len(meta) == 8, (len(bugs), len(meta))
    extra["bugs"] = sorted(bugs, key=lambda b: (b["file"], b["line"]))
    pathlib.Path(out_path).write_text(json.dumps(extra, indent=2) + "\n")
PY

# ------------------------------------------- chore/cleanup: 7 innocuous changes, no behaviour change
git checkout -q -b chore/cleanup
cat > "$TMP/a.py" <<'PY'
# 1. docs
edit("README.md", "payment capture.\n", "payment capture.\n\n## Running the tests\n\n"
     "```bash\npytest tests/\n```\n\nTests need `DATABASE_URL` only for the integration suite.\n")
# 2. docstring
edit("service/db.py", "def connect() -> psycopg.Connection:\n",
     'def connect() -> psycopg.Connection:\n    """Open an autocommit connection to the ShopFlow database."""\n')
# 3. refactor: extract a row-to-dict helper
edit("service/orders.py",
     '    return {"id": row[0], "customer_id": row[1], "status": row[2], "total_cents": row[3]}\n',
     "    return _row_to_order(row)\n\n\n"
     "def _row_to_order(row: tuple) -> dict:\n"
     '    return {"id": row[0], "customer_id": row[1], "status": row[2], "total_cents": row[3]}\n')
# 4. type hints + f-string
edit("service/orders.py", 'def format_money(cents):\n    return "$%.2f" % (cents / 100)\n',
     'def format_money(cents: int) -> str:\n    return f"${cents / 100:,.2f}"\n')
# 5. named constant
edit("service/inventory.py", "from .db import connect\n",
     "from .db import connect\n\nLOW_STOCK_THRESHOLD = 5\n")
edit("service/inventory.py", "def low_stock(threshold: int = 5)",
     "def low_stock(threshold: int = LOW_STOCK_THRESHOLD)")
# 6. logging
edit("service/payments.py", "from . import gateway, notify\n",
     "import logging\n\nfrom . import gateway, notify\n\nlogger = logging.getLogger(__name__)\n")
edit("service/payments.py", "    await notify.send_receipt(order_id)\n    return True\n",
     '    logger.info("captured order %s for %s cents", order_id, amount_cents)\n'
     "    await notify.send_receipt(order_id)\n    return True\n")
# 7. a new unit test
write("tests/test_orders.py",
      '"""Unit tests that need no database."""\n\nfrom service.orders import format_money\n\n\n'
      "def test_format_money_has_thousands_separator():\n"
      '    assert format_money(123456) == "$1,234.56"\n')
PY
run_edits "$TMP/a.py"
commit "Tidy the order service: docs, helper, hints, logging, test" "2026-10-27T10:00:00+00:00"

# --------------------------------- feature/order-fixes: chore/cleanup + 8 planted bugs
git checkout -q -b feature/order-fixes
cat > "$TMP/b.py" <<'PY'
# secret_in_code
edit("service/config.py", 'STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "")\n',
     'STRIPE_API_KEY = "4f9a1c7e0b2d4e68a3c5f1907d8b6e42"  #@@bug:secret_in_code\n')
# sql_injection
edit("service/orders.py",
     '        return conn.execute(\n'
     '            "SELECT id, customer_id, total_cents FROM orders WHERE status = %s ORDER BY id",\n'
     '            (status,),\n'
     '        ).fetchall()\n',
     '        return conn.execute(\n'
     '            f"SELECT id, customer_id, total_cents FROM orders WHERE status = \'{status}\' "  #@@bug:sql_injection\n'
     '            "ORDER BY id"  #@@end:sql_injection\n'
     '        ).fetchall()\n')
# off_by_one
edit("service/orders.py", "    offset = (page - 1) * PAGE_SIZE\n",
     "    offset = page * PAGE_SIZE  #@@bug:off_by_one\n")
# wrong_timezone
edit("service/orders.py", "from zoneinfo import ZoneInfo\n\nfrom .config import BUSINESS_TZ, PAGE_SIZE\n",
     "\nfrom .config import PAGE_SIZE\n")
edit("service/orders.py",
     "    tz = ZoneInfo(BUSINESS_TZ)\n"
     "    start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)\n",
     "    start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)  #@@bug:wrong_timezone\n")
# n_plus_one
edit("service/orders.py",
     "    with connect() as conn:\n"
     "        rows = conn.execute(\n"
     '            "SELECT customer_id, sum(total_cents) FROM orders "\n'
     '            "WHERE customer_id = ANY(%s) GROUP BY customer_id",\n'
     "            (customer_ids,),\n"
     "        ).fetchall()\n"
     "    totals = {cid: 0 for cid in customer_ids}\n"
     "    totals.update({cid: int(total) for cid, total in rows})\n"
     "    return totals\n",
     "    totals = {}\n"
     "    with connect() as conn:\n"
     "        for cid in customer_ids:\n"
     "            row = conn.execute(  #@@start:n_plus_one\n"
     '                "SELECT sum(total_cents) FROM orders WHERE customer_id = %s", (cid,)  #@@bug:n_plus_one\n'
     "            ).fetchone()  #@@end:n_plus_one\n"
     "            totals[cid] = int(row[0] or 0)\n"
     "    return totals\n")
edit("service/orders.py", 'Lifetime spend in cents per customer, in one query."""',
     'Lifetime spend in cents per customer."""')
# unchecked_none
edit("service/orders.py", "def format_money(cents: int) -> str:",
     "def describe_order(order_id: int) -> str:\n"
     '    """One-line summary for the dashboard header."""\n'
     "    order = get_order(order_id)\n"
     '    return f"Order {order[\'id\']} is {order[\'status\']} ({format_money(order[\'total_cents\'])})"  #@@bug:unchecked_none\n'
     "\n\n"
     "def format_money(cents: int) -> str:")
# race_condition
edit("service/inventory.py",
     "    with connect() as conn:\n"
     "        cur = conn.execute(\n"
     '            "UPDATE inventory SET on_hand = on_hand - %s "\n'
     '            "WHERE product_id = %s AND warehouse = %s AND on_hand >= %s",\n'
     "            (qty, product_id, warehouse, qty),\n"
     "        )\n"
     "        return cur.rowcount == 1\n",
     "    on_hand = available(product_id, warehouse)  #@@start:race_condition\n"
     "    if on_hand < qty:\n"
     "        return False\n"
     "    with connect() as conn:\n"
     "        conn.execute(\n"
     '            "UPDATE inventory SET on_hand = %s WHERE product_id = %s AND warehouse = %s",  #@@bug:race_condition\n'
     "            (on_hand - qty, product_id, warehouse),\n"
     "        )\n"
     "    return True\n")
# missing_await
edit("service/payments.py", "    await notify.send_receipt(order_id)\n",
     "    notify.send_receipt(order_id)  #@@bug:missing_await\n")

GT = {
    "base": "main",
    "clean_head": "chore/cleanup",
    "buggy_head": "feature/order-fixes",
    "line_refers_to": "line numbers in the head version of the file (new side of the diff); start..end is the region a reviewer may point at",
    "bugs": {
        "secret_in_code": {"categories": ["security"], "description": "Live Stripe key hard-coded instead of read from the environment"},
        "sql_injection": {"categories": ["security"], "description": "status is interpolated into SQL with an f-string"},
        "off_by_one": {"categories": ["correctness"], "description": "page is 1-based but offset = page * PAGE_SIZE skips the first page"},
        "wrong_timezone": {"categories": ["correctness"], "description": "'today' starts at naive UTC midnight, not America/New_York midnight"},
        "n_plus_one": {"categories": ["performance"], "description": "one query per customer inside a loop instead of one GROUP BY query"},
        "unchecked_none": {"categories": ["correctness"], "description": "get_order() may return None; describe_order indexes it without a check"},
        "race_condition": {"categories": ["concurrency", "correctness"], "description": "check-then-update on on_hand lets two buyers oversell; was one atomic UPDATE"},
        "missing_await": {"categories": ["correctness", "concurrency"], "description": "send_receipt() is a coroutine; without await it never runs"},
    },
    "innocuous": [
        {"file": "README.md", "description": "docs: how to run tests"},
        {"file": "service/db.py", "description": "docstring on connect()"},
        {"file": "service/orders.py", "description": "refactor: extract _row_to_order helper"},
        {"file": "service/orders.py", "description": "type hints and f-string in format_money"},
        {"file": "service/inventory.py", "description": "named constant LOW_STOCK_THRESHOLD"},
        {"file": "service/payments.py", "description": "logging line after a successful capture"},
        {"file": "tests/test_orders.py", "description": "new unit test for format_money"},
    ],
}
collect_markers(os.environ["GT_OUT"], GT)
PY
GT_OUT="$GT_OUT" run_edits "$TMP/b.py"
commit "Fix order lookups, stock reservation and receipts" "2026-11-03T10:00:00+00:00"

git checkout -q main
echo "sample repo: $DEST"
echo "  clean PR:  main...chore/cleanup  ($(git diff --shortstat main chore/cleanup))"
echo "  buggy PR:  main...feature/order-fixes  ($(git diff --shortstat main feature/order-fixes))"
echo "ground truth: $GT_OUT"
