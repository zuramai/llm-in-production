"""Capture a PNG of a page served by one of the book's apps.

    uv run playwright install chromium   # once
    uv run python tools/screenshot.py http://localhost:8001 ch01/images/01-chat-ui.png

Interact before capturing with ordered steps (each runs in the order given):

    uv run python tools/screenshot.py http://localhost:8001 out.png \\
        --fill "#input" "How long do I have to return an item?" --press Enter \\
        --wait-selector ".meta"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


class Step(argparse.Action):
    """Append (kind, *values) to args.steps, preserving command-line order."""

    def __call__(self, parser, namespace, values, option_string=None):
        steps = getattr(namespace, "steps", None) or []
        values = values if isinstance(values, list) else [values]
        steps.append((self.dest, *values))
        namespace.steps = steps


def main() -> None:
    p = argparse.ArgumentParser(description="Screenshot a URL with headless Chromium")
    p.add_argument("url")
    p.add_argument("out", type=Path)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--full-page", action="store_true")
    p.add_argument("--fill", nargs=2, metavar=("CSS", "TEXT"), action=Step,
                   help="type TEXT into the element")
    p.add_argument("--press", metavar="KEY", action=Step, help="press a key, e.g. Enter")
    p.add_argument("--click", metavar="CSS", action=Step, help="click the element")
    p.add_argument("--wait-selector", metavar="CSS", action=Step,
                   help="wait until a new element matching CSS appears")
    p.add_argument("--delay-ms", type=int, metavar="MS", action=Step, help="wait MS milliseconds")
    p.set_defaults(steps=[])
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(args.url, wait_until="networkidle")
        for kind, *values in args.steps:
            if kind == "fill":
                page.fill(values[0], values[1])
            elif kind == "press":
                page.keyboard.press(values[0])
            elif kind == "click":
                page.click(values[0])
            elif kind == "wait_selector":
                # Wait for one more match than exists now, so repeated waits work per message.
                count = page.locator(values[0]).count()
                page.wait_for_function(
                    "([sel, n]) => document.querySelectorAll(sel).length > n",
                    arg=[values[0], count], timeout=120_000)
            elif kind == "delay_ms":
                page.wait_for_timeout(int(values[0]))
        page.screenshot(path=str(args.out), full_page=args.full_page)
        browser.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
