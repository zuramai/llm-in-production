"""A ~30-line logging handler that pushes JSON log lines straight to Loki's push API.

Stdlib only. In production you would write to stdout and let Alloy or Promtail ship the lines.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request

FIELDS = ("route", "status", "duration_ms", "deploy_version", "error", "provider")


def build_push_payload(records: list[logging.LogRecord], service: str) -> dict:
    """Group records into one Loki stream per level. Timestamps are nanosecond strings."""
    streams: dict[str, list[list[str]]] = {}
    last_ts = 0
    for r in records:
        line = {"msg": r.getMessage()}
        line.update({k: v for k, v in getattr(r, "fields", {}).items() if k in FIELDS})
        ts = max(int(r.created * 1e9), last_ts + 1)  # strictly increasing, so no ties in a stream
        last_ts = ts
        streams.setdefault(r.levelname.lower(), []).append([str(ts), json.dumps(line)])
    return {"streams": [{"stream": {"service": service, "level": lv}, "values": v}
                        for lv, v in streams.items()]}


class LokiHandler(logging.Handler):
    def __init__(self, url: str, service: str = "checkout", flush_s: float = 1.0, max_queue=5000):
        super().__init__()
        self.url, self.service, self.flush_s = url, service, flush_s
        self.q: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put_nowait(record)  # never block a request on logging
        except queue.Full:
            self.dropped += 1

    def _loop(self) -> None:
        while True:
            time.sleep(self.flush_s)
            batch = []
            while not self.q.empty() and len(batch) < 500:
                batch.append(self.q.get_nowait())
            if batch:
                self._push(batch)

    def _push(self, batch: list[logging.LogRecord]) -> bool:
        body = json.dumps(build_push_payload(batch, self.service)).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=3).close()
            return True
        except Exception:  # noqa: BLE001 Loki down: drop the batch rather than crash or block the service
            self.dropped += len(batch)
            return False
