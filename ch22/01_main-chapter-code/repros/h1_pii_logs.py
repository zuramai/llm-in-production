"""H1: turn on DEBUG to chase a bug, and the SDK logs every prompt, PII included."""

from __future__ import annotations

import logging
import re
from http.server import BaseHTTPRequestHandler

import openai
from ch22_common import PORT, Result, completion_json, serve, timed

ID, TITLE, LAYER, KIND = "H1", "PII in logs", "Security", "local"

PROMPT = ("Hi, I'm Jane Roe, jane.roe.1987@gmail.com, card 4242 4242 4242 4242, phone "
          "+1 (917) 555-0143. Where is order 4512?")
PII = ["jane.roe.1987@gmail.com", "4242 4242 4242 4242", "555-0143"]

PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    (re.compile(r"\+?\d[\d ()-]{7,}\d"), "[PHONE]"),
]


def redact(text: str) -> str:
    for pattern, label in PATTERNS:
        text = pattern.sub(label, text)
    return text


class RedactingFilter(logging.Filter):
    """Redact the *formatted* message, so PII inside args (dicts, request bodies) is caught too."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        # The model politely repeats the customer's details back: PII on the output side too.
        body = completion_json("Thanks Jane! I'll send updates on order 4512 to "
                               "jane.roe.1987@gmail.com.")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


app_log = logging.getLogger("shopflow.chat")


def handle_chat(client: openai.OpenAI, message: str) -> str:
    """A chat handler with the logging line every team adds while debugging."""
    r = client.chat.completions.create(model="stub", max_tokens=20,
                                       messages=[{"role": "user", "content": message}])
    reply = r.choices[0].message.content
    app_log.info("chat turn: user=%r reply=%r", message, reply)
    return reply


def logged_lines(redacting: bool) -> dict[str, list[str]]:
    """Capture SDK/HTTP debug logs and app logs separately for one chat turn."""
    sdk, app = Capture(), Capture()
    if redacting:
        for cap in (sdk, app):
            cap.addFilter(RedactingFilter())
    wiring = [(logging.getLogger(n), sdk) for n in ("openai", "httpx", "httpcore")]
    wiring.append((app_log, app))
    old = [(lg.level, lg.propagate) for lg, _ in wiring]
    for lg, cap in wiring:
        lg.setLevel(logging.DEBUG)
        lg.propagate = False
        lg.addHandler(cap)
    try:
        client = openai.OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="stub",
                               max_retries=0)
        handle_chat(client, PROMPT)
    finally:
        for (lg, cap), (level, prop) in zip(wiring, old, strict=True):
            lg.removeHandler(cap)
            lg.setLevel(level)
            lg.propagate = prop
    return {"sdk": sdk.lines, "app": app.lines}


def _leaks(lines: list[str]) -> list[str]:
    return [p for p in PII if any(p in line for line in lines)]


@timed
def run() -> Result:
    with serve(Handler):
        naive = logged_lines(redacting=False)
        fixed = logged_lines(redacting=True)
    sample = fixed["app"][0] if fixed["app"] else ""
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=bool(_leaks(naive["app"])),
        fixed=not _leaks(fixed["app"] + fixed["sdk"]) and "[EMAIL]" in sample,
        evidence=f"openai {openai.__version__} + httpx at DEBUG: {len(naive['sdk'])} lines, PII "
                 f"{_leaks(naive['sdk']) or 'none'}; app's own log line: PII "
                 f"{_leaks(naive['app'])}; with RedactingFilter: "
                 f"{_leaks(fixed['app'] + fixed['sdk']) or 'none'} -> {sample[:150]!r}",
        data={"naive": naive, "fixed": fixed},
    )
