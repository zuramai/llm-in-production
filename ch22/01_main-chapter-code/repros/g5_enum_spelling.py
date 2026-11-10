"""G5: `status <> 'canceled'` when the data says 'cancelled': a filter that silently does nothing."""

from __future__ import annotations

import sqlglot
from ch22_common import Result, timed
from sqlglot import exp

from llmprod import connect

ID, TITLE, LAYER, KIND = "G5", "Silent no-op filter: 'canceled' vs 'cancelled'", "Data & time", "db"

# What Chapter 6's v0 bot generated, minus the time filter (US spelling; the data is British).
GENERATED = "SELECT count(*) FROM orders WHERE status <> 'canceled'"
CORRECT = "SELECT count(*) FROM orders WHERE status <> 'cancelled'"


class UnknownEnumValue(ValueError):
    pass


def check_enum_literals(sql: str, allowed: dict[str, set[str]]) -> None:
    """Reject comparisons of a known enum column with a value that doesn't exist in the data."""
    for node in sqlglot.parse_one(sql, read="postgres").find_all(
            exp.EQ, exp.NEQ, exp.In):
        col = node.this if isinstance(node.this, exp.Column) else None
        if col is None or col.name not in allowed:
            continue
        values = node.expressions if isinstance(node, exp.In) else [node.expression]
        for v in values:
            if isinstance(v, exp.Literal) and v.is_string and v.this not in allowed[col.name]:
                raise UnknownEnumValue(
                    f"{col.name} compared with {v.this!r}, a value it never has; known values: "
                    f"{sorted(allowed[col.name])}")


@timed
def run() -> Result:
    with connect(readonly=True) as conn:
        total = conn.execute("SELECT count(*) FROM orders").fetchone()[0]
        naive = conn.execute(GENERATED).fetchone()[0]
        right = conn.execute(CORRECT).fetchone()[0]
        statuses = {r[0] for r in conn.execute("SELECT DISTINCT status FROM orders")}
    try:
        check_enum_literals(GENERATED, {"status": statuses})
        caught = None
    except UnknownEnumValue as e:
        caught = str(e)
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=naive == total and right < total,
        fixed=caught is not None,
        evidence=f"{total:,} orders; `status <> 'canceled'` -> {naive:,} (no error, filters "
                 f"nothing); 'cancelled' -> {right:,}. Literal check: {caught}",
    )
