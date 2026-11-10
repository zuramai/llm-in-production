"""Static checks on LLM-generated SQL before it reaches the database.

The guard is one layer of defense; the read-only database role is another. Every error
message names the rule that was broken, because the message is fed back to the model.

Copied unchanged from ch06/01_main-chapter-code/sqlbot/guard.py.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

BLOCKED_FUNCTIONS = frozenset({
    "pg_sleep", "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "dblink", "dblink_exec", "copy", "pg_terminate_backend",
    "pg_cancel_backend", "set_config", "pg_reload_conf",
})
WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Command, exp.Copy, exp.Grant,
)


class UnsafeSQLError(ValueError):
    """The SQL breaks a guard rule. The message is safe to show to the model."""


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return node.name.lower()
    return node.sql_name().lower()


def validate_sql(sql: str, allowed_tables: set[str], max_limit: int = 1000) -> str:
    """Return normalized, LIMIT-capped Postgres SQL, or raise UnsafeSQLError."""
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="postgres") if s is not None]
    except ParseError as e:
        raise UnsafeSQLError(f"rule=parse: SQL does not parse as Postgres: {e}") from None
    if len(statements) != 1:
        raise UnsafeSQLError(
            f"rule=single_statement: expected exactly 1 statement, got {len(statements)}")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        raise UnsafeSQLError(
            f"rule=select_only: only SELECT queries are allowed, got {tree.key.upper()}")
    for node in tree.walk():
        if isinstance(node, WRITE_NODES):
            raise UnsafeSQLError(
                f"rule=select_only: {node.key.upper()} is not allowed anywhere in the query")
        if isinstance(node, exp.Select) and node.args.get("into"):
            raise UnsafeSQLError("rule=no_select_into: SELECT ... INTO creates a table")
        if isinstance(node, exp.Func) and _function_name(node) in BLOCKED_FUNCTIONS:
            raise UnsafeSQLError(
                f"rule=blocked_function: function {_function_name(node)}() is not allowed")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables} | cte_names
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        schema = table.db.lower()
        if not name or schema not in ("", "public") or name not in allowed:
            shown = f"{schema}.{name}" if schema else (name or table.sql())
            raise UnsafeSQLError(
                f"rule=allowed_tables: table {shown} is not allowed; "
                f"use only: {', '.join(sorted(allowed_tables))}")

    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(max_limit)
    else:
        value = limit.expression
        if not (isinstance(value, exp.Literal) and value.is_int):
            raise UnsafeSQLError("rule=limit: LIMIT must be an integer literal")
        if int(value.this) > max_limit:
            tree = tree.limit(max_limit)
    return tree.sql(dialect="postgres")
