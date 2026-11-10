import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch18_data import CATEGORIES, DATA, read_jsonl
from ch18_prepare import STYLES, fill_order_ids, synth_jobs


def test_synth_jobs_balance_categories_and_cycle_styles():
    jobs = synth_jobs(30, 10)
    cats = collections.Counter(c for j in jobs for c, _ in j["targets"])
    assert set(cats) == set(CATEGORIES) and set(cats.values()) == {50}
    assert [j["style"] for j in jobs[: len(STYLES)]] == STYLES
    assert synth_jobs(30, 10) == jobs  # seeded


def test_fill_order_ids_uses_one_number_for_subject_and_body():
    s, b, oid = fill_order_ids("Order #{order_id} late, #{order_id}!", "About #{order_id}",
                               random.Random(1))
    assert oid is not None and "{order_id}" not in s + b
    assert b.count(str(oid)) == 2 and str(oid) in s


def test_fill_order_ids_leaves_tickets_without_an_order_alone():
    assert fill_order_ids("no order here", "hi", random.Random(1)) == ("hi", "no order here", None)


def test_committed_synthetic_rows_are_clean():
    rows = read_jsonl(DATA / "synthetic.jsonl")
    assert len(rows) > 200
    assert all(r["max_sim_to_hard_test"] < 0.5 for r in rows)  # no near-copies of the exam
    assert all("{order_id}" not in r["subject"] + r["body"] for r in rows)
    assert all((r["order_id"] is None) == (str(r["order_id"]) not in r["body"] + r["subject"])
               for r in rows)
    test_ids = {r["id"] for r in read_jsonl(DATA / "test.jsonl")}
    assert not test_ids & {r["id"] for r in rows}
