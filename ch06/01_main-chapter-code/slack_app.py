"""ShopFlow data bot for Slack (Socket Mode: no public URL needed).

    cd ch06/01_main-chapter-code
    uv run python slack_app.py          # needs SLACK_BOT_TOKEN and SLACK_APP_TOKEN in .env
"""

from __future__ import annotations

import os
import re

from dotenv import find_dotenv, load_dotenv
from sqlbot import answer

MENTION = re.compile(r"<@[A-Z0-9]+>\s*")


def handle_question(text: str) -> list[dict]:
    """Answer `text` and return Slack Block Kit blocks. Pure: no Slack API calls."""
    question = MENTION.sub("", text).strip()
    if not question:
        return [{"type": "section",
                 "text": {"type": "mrkdwn", "text": "Ask me a question about ShopFlow's data, "
                                                    "e.g. _How many transactions are there today?_"}}]
    try:
        result = answer(question)
    except Exception as exc:  # noqa: BLE001 — the bot must always reply, whatever broke
        return [{"type": "section",
                 "text": {"type": "mrkdwn", "text": f"Sorry, I couldn't answer that: {exc}"}}]
    blocks: list[dict] = [{"type": "section", "text": {"type": "mrkdwn", "text": result.answer}}]
    details = f"{result.attempts} attempt(s) · {result.latency_ms / 1000:.1f} s · {result.model}"
    if result.sql:
        sql = " ".join(result.sql.split())
        sql = sql if len(sql) <= 2500 else sql[:2500] + "…"
        details = f"`{sql}`\n{details}"
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": details}]})
    return blocks


def create_app():
    from slack_bolt import App

    app = App(token=os.environ["SLACK_BOT_TOKEN"])

    def respond(event: dict, client, say) -> None:
        channel, ts = event["channel"], event["ts"]
        client.reactions_add(channel=channel, timestamp=ts, name="hourglass_flowing_sand")
        try:
            blocks = handle_question(event.get("text", ""))
            fallback = blocks[0]["text"]["text"]
            say(blocks=blocks, text=fallback, thread_ts=event.get("thread_ts", ts))
        finally:
            client.reactions_remove(channel=channel, timestamp=ts, name="hourglass_flowing_sand")

    @app.event("app_mention")
    def on_mention(event, client, say):
        respond(event, client, say)

    @app.event("message")
    def on_dm(event, client, say):
        # Only direct messages from humans; mentions in channels arrive as app_mention.
        if event.get("channel_type") == "im" and not event.get("bot_id") \
                and not event.get("subtype"):
            respond(event, client, say)

    return app


def main() -> None:
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    load_dotenv(find_dotenv(usecwd=True))
    missing = [v for v in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN") if not os.environ.get(v)]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)} in .env (see ch06/README.md, Serving it)")
    SocketModeHandler(create_app(), os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
