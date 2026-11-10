"""F4: tiktoken downloads its BPE file on first use, so an offline container crashes at startup."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tiktoken
from ch22_common import Result, timed

ID, TITLE, LAYER, KIND = "F4", "tiktoken downloads at runtime: offline crash", "Infra & serving", \
    "local"

SNIPPET = "import tiktoken; print(len(tiktoken.get_encoding('o200k_base').encode('hello world')))"
URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
# An unreachable proxy stands in for an air-gapped cluster with no egress.
OFFLINE = {"HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
           "NO_PROXY": ""}


def python(env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
    return subprocess.run([sys.executable, "-c", SNIPPET], env=env | env_extra,
                          capture_output=True, text=True, timeout=120, check=False)


def bake(cache_dir: str) -> None:
    """Build step (has network): put the BPE file into the dir that ships inside the image.

    Equivalent to `RUN TIKTOKEN_CACHE_DIR=/opt/tiktoken python -c "...get_encoding(...)"` in a
    Dockerfile. We copy from this host's cache because the download itself is slow here.
    """
    tiktoken.get_encoding("o200k_base")  # downloads into the default cache if missing
    name = hashlib.sha1(URL.encode()).hexdigest()
    src = Path(os.environ.get("TIKTOKEN_CACHE_DIR")
               or os.environ.get("DATA_GYM_CACHE_DIR")
               or Path(tempfile.gettempdir()) / "data-gym-cache") / name
    shutil.copy(src, Path(cache_dir) / name)


@timed
def run() -> Result:
    with tempfile.TemporaryDirectory() as empty, tempfile.TemporaryDirectory() as baked:
        naive = python({"TIKTOKEN_CACHE_DIR": empty} | OFFLINE)
        bake(baked)
        fixed = python({"TIKTOKEN_CACHE_DIR": baked} | OFFLINE)
        files = len(os.listdir(baked))
    last = (naive.stderr.strip().splitlines() or [""])[-1]
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=naive.returncode != 0,
        fixed=fixed.returncode == 0 and fixed.stdout.strip() == "2",
        evidence=f"no egress, empty TIKTOKEN_CACHE_DIR -> exit {naive.returncode}: {last[:90]}; "
                 f"cache baked at build time ({files} file) -> exit {fixed.returncode}, "
                 f"'hello world' = {fixed.stdout.strip()} tokens",
    )
