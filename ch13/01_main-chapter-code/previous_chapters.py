"""Code from earlier chapters that Chapter 13 attacks, copied so the chapter stands alone.

The copies live in their own uniquely named modules (tests import them, and test module names
must be unique across the repo); this file is the index of what came from where.

| ch13 module           | copied from                                         | ch13 changes                        |
|-----------------------|-----------------------------------------------------|-------------------------------------|
| ch13_sqlbot/agent.py  | ch06/01_main-chapter-code/sqlbot/agent.py           | canary line; `conninfo` argument    |
| ch13_sqlbot/db.py     | ch06/01_main-chapter-code/sqlbot/db.py              | `conninfo` (connect as another role)|
| ch13_sqlbot/guard.py  | ch06/01_main-chapter-code/sqlbot/guard.py           | none                                |
| ch13_sqlbot/semantic_context.yaml | ch06 (same path)                        | tickets -> `ch13_support_tickets`, `body` documented |
| ch13_agent.py         | ch09/01_main-chapter-code/ch09_agent.py             | canary line; `system_for(role)`     |
| ch13_tools.py         | ch09/01_main-chapter-code/ch09_tools.py             | + `get_ticket`, `search_docs`       |
| ch13_refunds.py       | ch09/01_main-chapter-code/ch09_refunds.py           | request side only; own table        |
"""

from ch13_agent import CANARY, run_agent, system_for
from ch13_sqlbot import answer as sqlbot_answer
from ch13_tools import LocalToolbox

__all__ = ["CANARY", "LocalToolbox", "run_agent", "sqlbot_answer", "system_for"]
