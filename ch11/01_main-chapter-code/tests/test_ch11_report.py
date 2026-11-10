import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch11_agent.report import (
    Hypothesis,
    IncidentReport,
    ranked,
    render_markdown,
    response_format,
    strict_schema,
)
from ch11_agent.scoring import matches, rank_of_root_cause


def report(*hyps):
    return IncidentReport(summary="s", impact="i", timeline=["10:00 deploy v2"],
                          hypotheses=[Hypothesis(hypothesis=h, evidence=["e1"], confidence=c)
                                      for h, c in hyps],
                          next_actions=["Roll back to v1"])


def test_strict_schema_is_closed_and_fully_required_at_every_level():
    schema = strict_schema(IncidentReport)

    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object":
                assert n["additionalProperties"] is False
                assert set(n["required"]) == set(n["properties"])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(schema)
    assert "Hypothesis" in schema["$defs"]
    assert response_format()["json_schema"]["strict"] is True


def test_ranked_sorts_by_confidence_clamps_and_is_stable():
    r = report(("a", 0.4), ("b", 1.7), ("c", 0.4), ("d", -3))
    out = ranked(r)
    assert [h.hypothesis for h in out] == ["b", "a", "c", "d"]
    assert out[0].confidence == 1.0 and out[-1].confidence == 0.0


def test_render_markdown_orders_hypotheses_and_labels_actions_as_proposals():
    r = report(("minor", 0.2), ("DB pool exhausted", 0.9))
    md = render_markdown(r, alert="5xx high", model="m", steps=4, cost_usd=0.01234,
                         generated_at=datetime(2026, 11, 7, 12, 0, tzinfo=UTC))
    assert md.index("### 1. DB pool exhausted (confidence 90%)") < md.index("### 2. minor")
    assert "a human decides and executes" in md
    assert "1. Roll back to v1" in md
    assert "2026-11-07 12:00 UTC" in md and "$0.0123" in md


def test_scoring_requires_all_groups_and_the_right_provider():
    assert matches("Slow paypal timeouts", "payment-provider-timeout", "paypal")
    assert not matches("Slow stripe timeouts", "payment-provider-timeout", "paypal")
    assert not matches("PayPal is fine", "payment-provider-timeout", "paypal")
    assert matches("DB connection pool exhausted", "db-pool-exhaustion")
    assert not matches("Bad deploy of v2", "db-pool-exhaustion")


@pytest.mark.parametrize(("hyps", "rank"), [
    ([("Bad deploy v2 broke checkout", 0.9), ("DB pool exhausted", 0.3)], 2),
    ([("Bad deploy v2 broke checkout", 0.3), ("DB pool exhausted", 0.9)], 1),
    ([("Network blip", 0.9)], None),
])
def test_rank_of_root_cause_uses_confidence_order_not_list_order(hyps, rank):
    assert rank_of_root_cause(report(*hyps), "db-pool-exhaustion") == rank


def test_bad_deploy_counts_the_named_culprit_and_rejects_the_distractor_release():
    bad = "v2026.11.07-2"
    # the real raw-tools run that the first keyword rule scored as a miss
    assert matches("A bug in v2026.11.07-2's promo-pricing change causes /checkout exceptions",
                   "bad-deploy", bad_version=bad)
    assert not matches("Deploy v2026.11.07-1 (gift-message field) broke checkout", "bad-deploy",
                       bad_version=bad)
    assert matches("Bad deploy of the promo-pricing change", "bad-deploy", bad_version=bad)


@pytest.mark.parametrize(("text", "scenario", "hit"), [
    # real gpt-oss:20b top hypotheses, 2026-11-09, judged by reading the full reports
    (("New discount logic introduced a null reference when applying promo codes, causing "
      "TypeError in pricing calculation."), "bad-deploy", True),
    ("Bad deploy introduced a None dereference in /checkout.", "bad-deploy", True),
    (("The recent deploy introduced a new code path that increases database contention, leading "
      "to higher pool wait times and downstream request timeouts manifested as 5xx errors."),
     "db-pool-exhaustion", False),
    ("Database connection pool exhaustion due to increased load after deployment.",
     "db-pool-exhaustion", False),
    ("Database connection pool exhaustion caused 503 responses", "db-pool-exhaustion", True),
    (("New gift\u2011message field in checkout payload caused malformed requests to PayPal, "
      "leading to timeouts."), "payment-provider-timeout", False),
])
def test_the_injected_release_counts_without_deploy_and_blaming_a_healthy_one_does_not(
        text, scenario, hit):
    assert matches(text, scenario, bad_version="v2026.11.09-2") is hit


def test_right_mechanism_blamed_on_an_innocent_release_does_not_count():
    # a real claude-haiku-5.5 top hypothesis in the db-pool scenario; it proposed rolling back
    blames_distractor = ("Deploy v2026.11.07-1 (gift-message change) causes DB connection pool "
                         "exhaustion")
    assert not matches(blames_distractor, "db-pool-exhaustion")
    assert matches("Checkout DB connection pool exhausted", "db-pool-exhaustion")
    assert rank_of_root_cause(report((blames_distractor, 0.85), ("DB pool exhausted", 0.5)),
                              "db-pool-exhaustion") == 2
