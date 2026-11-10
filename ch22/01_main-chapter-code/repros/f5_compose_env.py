"""F5: Docker Compose interpolation: your shell beats --env-file; `${VAR:?}` breaks `down`."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from ch22_common import Result, timed

ID, TITLE, LAYER, KIND = "F5", "Compose env precedence and ${VAR:?} on down", "Infra & serving", \
    "docker"

NAIVE = """services:
  web:
    image: busybox
    environment:
      DATABASE_URL: ${DATABASE_URL:-postgresql://langfuse@db:5432/langfuse}
      LLM_API_KEY: ${LLM_API_KEY:?set LLM_API_KEY}
"""
FIXED = """services:
  web:
    image: busybox
    environment:
      DATABASE_URL: ${LANGFUSE_DATABASE_URL:-postgresql://langfuse@db:5432/langfuse}
      LLM_API_KEY: ${LLM_API_KEY:?set LLM_API_KEY}
"""
ENV_FILE = "DATABASE_URL=postgresql://langfuse@db:5432/langfuse\n"
# What the shell of someone who sourced this book's .env looks like:
SHELL = {"DATABASE_URL": "postgresql://shopflow@localhost:5433/shopflow"}


def compose(workdir: Path, compose_file: str | None, *args: str,
            env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "LLM_API_KEY",
                                                           "LANGFUSE_DATABASE_URL")}
    cmd = ["docker", "compose", "-p", "ch22-f5"]
    if compose_file:
        cmd += ["-f", compose_file, "--env-file", "svc.env"]
    return subprocess.run(cmd + list(args), cwd=workdir, env=env | env_extra,
                          capture_output=True, text=True, timeout=60, check=False)


def resolved_db_url(workdir: Path, compose_file: str) -> str:
    r = compose(workdir, compose_file, "config", "--format", "json",
                env_extra=SHELL | {"LLM_API_KEY": "sk-test"})
    r.check_returncode()
    return json.loads(r.stdout)["services"]["web"]["environment"]["DATABASE_URL"]


@timed
def run() -> Result:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "naive.yaml").write_text(NAIVE)
        (d / "fixed.yaml").write_text(FIXED)
        (d / "svc.env").write_text(ENV_FILE)
        naive_url = resolved_db_url(d, "naive.yaml")
        fixed_url = resolved_db_url(d, "fixed.yaml")
        down_naive = compose(d, "naive.yaml", "down", env_extra={})  # key not in this shell
        down_fixed = compose(d, None, "down", env_extra={})  # by project name, no file
    err = (down_naive.stderr.strip().splitlines() or [""])[-1]
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced="shopflow" in naive_url and down_naive.returncode != 0,
        fixed="langfuse" in fixed_url and down_fixed.returncode == 0,
        evidence=f"shell has DATABASE_URL, --env-file has the service's: ${{DATABASE_URL}} -> "
                 f"{naive_url}; ${{LANGFUSE_DATABASE_URL}} -> {fixed_url}. `down` without "
                 f"LLM_API_KEY -> exit {down_naive.returncode} ({err[:60]}); `docker compose -p "
                 f"ch22-f5 down` -> exit {down_fixed.returncode}",
    )
