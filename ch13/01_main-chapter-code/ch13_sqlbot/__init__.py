"""Copy of the Chapter 6 text-to-SQL bot, used as an attack target in Chapter 13."""

from .agent import SqlAnswer, SqlPlan, answer
from .db import QueryError, run_query
from .guard import UnsafeSQLError, validate_sql

__all__ = [
    "QueryError",
    "SqlAnswer",
    "SqlPlan",
    "UnsafeSQLError",
    "answer",
    "run_query",
    "validate_sql",
]
