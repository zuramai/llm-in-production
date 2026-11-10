"""G3: strftime('%p') is locale data: under id_ID it's empty, so '5:19 PM' prints as '5:19 '."""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys

from ch22_common import Result, timed

ID, TITLE, LAYER, KIND = "G3", "strftime('%p') empty under a non-English locale", "Data & time", \
    "local"

# A plain Python process formats a time. No setlocale() call: on this host CPython already
# formats with LC_TIME from the environment.
SNIPPET = """
import datetime as dt, sys
sys.path.insert(0, {path!r})
from repros.g3_strftime_locale import as_of
t = dt.datetime(2026, 11, 7, 17, 19)
print(repr(t.strftime("as of %-I:%M %p")) + "|" + repr(as_of(t)))
"""


def as_of(t: dt.datetime, tz_abbrev: str = "") -> str:
    """'as of 5:19 PM EST', independent of LC_TIME (AM/PM built from the hour)."""
    hour = t.hour % 12 or 12
    suffix = "AM" if t.hour < 12 else "PM"
    return f"as of {hour}:{t.minute:02d} {suffix}" + (f" {tz_abbrev}" if tz_abbrev else "")


def in_locale(loc: str) -> tuple[str, str]:
    """Run SNIPPET with this host's login locale (LANG=en_US.UTF-8, LC_CTYPE=UTF-8, regional
    LC_* = id_ID.UTF-8) but LC_TIME set to `loc`."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LC_", "LANG"))}
    env |= {"LANG": "en_US.UTF-8", "LC_CTYPE": "UTF-8", "LC_TIME": loc}
    out = subprocess.run([sys.executable, "-c", SNIPPET.format(path=here)], env=env,
                         capture_output=True, text=True, timeout=60, check=True).stdout.strip()
    naive, fixed = out.split("|")
    return naive, fixed


@timed
def run() -> Result:
    res = {loc: in_locale(loc) for loc in ("C.UTF-8", "id_ID.UTF-8")}
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=res["id_ID.UTF-8"][0] != res["C.UTF-8"][0],
        fixed=len({fixed for _, fixed in res.values()}) == 1,
        evidence="; ".join(f"LC_TIME={k}: strftime -> {v[0]}, as_of() -> {v[1]}"
                           for k, v in res.items()),
        data=res,
    )
