"""Screenshot Langfuse pages after logging in (tools/screenshot.py can't keep a session).

    uv run python ch14/01_main-chapter-code/ch14_screenshots.py \\
        /project/sqlbot/traces/<trace_id>=ch14/images/14-trace.png

Credentials are the headless-init user from langfuse/langfuse.env.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ch14_obs as obs
from playwright.sync_api import sync_playwright

EMAIL = "admin@shopflow.local"
PASSWORD = "ch14-langfuse-admin"  # local demo value from langfuse/langfuse.env


def capture(shots: dict[str, Path], *, width: int = 1440, height: int = 900,
            settle_ms: int = 6000, base_url: str | None = None) -> None:
    """Log in once, then save one PNG per path in `shots` ({"/project/...": out_png})."""
    base = (base_url or obs.langfuse_settings()["base_url"]).rstrip("/")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(f"{base}/auth/sign-in", wait_until="networkidle")
        page.fill("input[name=email]", EMAIL)
        page.fill("input[name=password]", PASSWORD)
        page.keyboard.press("Enter")
        page.wait_for_url(lambda url: "/auth/" not in url, timeout=30_000)
        for path, out in shots.items():
            out.parent.mkdir(parents=True, exist_ok=True)
            page.goto(base + path, wait_until="networkidle")
            page.wait_for_timeout(settle_ms)
            page.screenshot(path=str(out))
            print(f"saved {out}")
        browser.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("shots", nargs="+", metavar="PATH=OUT.png")
    args = p.parse_args()
    # Split on the LAST "=": Langfuse URLs carry query strings (?...=...), file names don't.
    pairs = [s.rsplit("=", 1) for s in args.shots]
    capture({path: Path(out) for path, out in pairs})


if __name__ == "__main__":
    main()
