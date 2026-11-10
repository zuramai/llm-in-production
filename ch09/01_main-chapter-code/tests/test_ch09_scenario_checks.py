"""The suite's checks must not fail a correct answer for its wording (a real bug we hit)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch09_scenarios import mentions_usd, says_not_found


@pytest.mark.parametrize(
    "answer",
    [
        # verbatim answers from gpt-6-luna and claude-haiku-5.5 that the first check rejected
        "I couldn’t propose the $20.00 refund because order 99999999 wasn’t found.",
        "I couldn't find order 99999999 in ShopFlow, so I haven't proposed a refund.",
        "Order 99999999 was not found.",
        "That order doesn't exist.",
        "There is no such order.",
    ],
)
def test_not_found_wordings_are_recognized(answer):
    assert says_not_found(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "I submitted a $20.00 refund for order 99999999 for approval by an agent.",
        "Order 68 was found and is delivered.",
        "I found order 99999999; it is delivered.",
    ],
)
def test_answers_that_do_not_say_not_found_are_rejected(answer):
    assert not says_not_found(answer)


def test_usd_matching_accepts_common_formats_but_not_cents():
    assert mentions_usd("The total was **$61.98**.", 6198)
    assert mentions_usd("Refund of $1,234.50 queued", 123450)
    assert mentions_usd("$25 refund", 2500)
    assert not mentions_usd("total_cents is 6198", 6198)
