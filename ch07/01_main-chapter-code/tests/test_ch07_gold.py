"""The compiler and the hand-written gold SQL must agree on every answerable eval question."""

import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from ch07_semantic import MetricQuery, compile_query, default_model
from ch07_semantic.db import jsonable, run_query
from ch07_semantic.scoring import rows_match

from llmprod import connect

CANONICAL = yaml.safe_load((HERE / "eval" / "canonical.yaml").read_text())


def _gold() -> dict[str, str]:
    tz = default_model().timezone
    items = []
    for name in ("questions.yaml", "questions_metrics.yaml"):
        items += yaml.safe_load((HERE / "eval" / name).read_text().replace("{tz}", tz))
    return {i["id"]: i["gold_sql"] for i in items if "gold_sql" in i}


GOLD = _gold()


def test_every_answerable_question_has_a_canonical_query():
    assert set(CANONICAL) == set(GOLD)


@pytest.mark.docker
@pytest.mark.parametrize("item_id", sorted(CANONICAL))
def test_compiled_rows_equal_gold_rows(item_id):
    with connect(readonly=True) as conn:
        gold = [tuple(jsonable(v) for v in row) for row in conn.execute(GOLD[item_id])]
    rows = run_query(compile_query(MetricQuery(**CANONICAL[item_id])))
    assert rows_match(gold, rows), (gold[:3], rows[:3])
