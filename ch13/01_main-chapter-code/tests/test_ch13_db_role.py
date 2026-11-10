import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module")
def conninfo():
    import ch13_setup

    ch13_setup.setup()
    return ch13_setup.analytics_conninfo()


@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("SELECT full_name FROM customers LIMIT 1", "permission denied"),
        ("SELECT city FROM customers LIMIT 1", "permission denied"),
        ("SELECT email FROM customers LIMIT 1", "permission denied"),
        ("SELECT * FROM customers LIMIT 1", "permission denied"),
        ("DELETE FROM orders WHERE id = -1", "read-only transaction"),
    ],
)
def test_analytics_role_cannot_read_people_or_write(conninfo, sql, error):
    from ch13_sqlbot import QueryError, run_query

    with pytest.raises(QueryError, match=error):
        run_query(sql, conninfo)


def test_analytics_role_still_answers_aggregates(conninfo):
    from ch13_sqlbot import run_query

    rows = run_query("SELECT segment, count(*) AS n FROM customers GROUP BY segment", conninfo)
    assert {r["segment"] for r in rows} == {"consumer", "business", "vip"}
    rows = run_query("SELECT count(*) AS n FROM ch13_support_tickets WHERE id > 900000", conninfo)
    assert rows == [{"n": 6}]
