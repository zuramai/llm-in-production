"""Operations helpers for Chapter 20: database setup, DSN rewriting, shell and kind/kubectl calls.

Used by the notebook and by `deploy_kind.sh`-style steps. Secrets are read from the repo `.env`
and passed to kubectl on stdin; they are never printed.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql

from llmprod import get_settings

HERE = Path(__file__).resolve().parent
SERVICES = HERE / "services"
K8S = HERE / "k8s"
CLUSTER = "ch20"
NAMESPACE = "shopflow"
APP_ROLE = "ch20_app"


def app_db_password() -> str:
    # A local demo database. In a real deployment this comes from your secret manager.
    return os.environ.get("CH20_APP_DB_PASSWORD", "ch20-app-local-only")


def setup_db() -> None:
    """Idempotently create schema ch20, the session table and a least-privilege app role."""
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS ch20")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ch20.chat_sessions (
                session_id text PRIMARY KEY,
                messages   jsonb NOT NULL,
                updated_at timestamptz NOT NULL DEFAULT now()
            )""")
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s",
                              (APP_ROLE,)).fetchone()
        verb = "ALTER" if exists else "CREATE"
        conn.execute(sql.SQL(verb + " ROLE {} LOGIN PASSWORD {}").format(
            sql.Identifier(APP_ROLE), sql.Literal(app_db_password())))
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA ch20 TO {}").format(sql.Identifier(APP_ROLE)))
        conn.execute(sql.SQL(
            "GRANT SELECT, INSERT, UPDATE, DELETE ON ch20.chat_sessions TO {}").format(
            sql.Identifier(APP_ROLE)))


def with_host(dsn: str, host: str) -> str:
    """Return `dsn` with its hostname replaced (port, user, password and path kept)."""
    parts = urlsplit(dsn)
    userinfo, _, _ = parts.netloc.rpartition("@")
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{userinfo}@{host}{port}" if userinfo else f"{host}{port}"
    return urlunsplit(parts._replace(netloc=netloc))


def sessions_dsn(host: str = "localhost") -> str:
    base = get_settings().database_url
    parts = urlsplit(base)
    netloc = f"{APP_ROLE}:{app_db_password()}@{parts.hostname}:{parts.port or 5432}"
    return with_host(urlunsplit(parts._replace(netloc=netloc)), host)


def sh(cmd: str, *, check: bool = True, input: str | None = None, timeout: int = 600,
       env: dict[str, str] | None = None) -> str:
    """Run a shell command; return stdout+stderr. Raise with the output on failure."""
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, input=input,
                          timeout=timeout, check=False, env=env)
    out = (proc.stdout + proc.stderr).strip()
    if check and proc.returncode != 0:
        raise RuntimeError(f"`{cmd}` exited {proc.returncode}:\n{out}")
    return out


# The kube context to talk to: the kind cluster by default; set CH20_KUBE_CONTEXT="" in CI to
# use whatever context the KUBECONFIG selects.
KUBE_CONTEXT = os.environ.get("CH20_KUBE_CONTEXT", f"kind-{CLUSTER}")


def kubectl_cluster(args: str, **kw) -> str:
    context = f"--context {KUBE_CONTEXT} " if KUBE_CONTEXT else ""
    return sh(f"kubectl {context}{args}", **kw)


def kubectl(args: str, **kw) -> str:
    """kubectl in the shopflow namespace."""
    return kubectl_cluster(f"-n {NAMESPACE} {args}", **kw)


def kind_gateway() -> str:
    """IP of the host as seen from kind nodes (the `kind` Docker network gateway)."""
    out = sh("docker network inspect kind -f '{{range .IPAM.Config}}{{.Gateway}} {{end}}'")
    return next(ip for ip in out.split() if "." in ip)


def apply_secret(name: str, data: dict[str, str]) -> str:
    """Create or update a Secret from `data` without the values touching argv or the output."""
    # json.dumps gives a double-quoted string, which is valid YAML for any value.
    manifest_lines = [
        "apiVersion: v1", "kind: Secret", "metadata:", f"  name: {name}",
        f"  namespace: {NAMESPACE}", "type: Opaque", "stringData:",
        *(f"  {k}: {json.dumps(v)}" for k, v in data.items()),
    ]
    return kubectl("apply -f -", input="\n".join(manifest_lines) + "\n")


def wait_rollout(deployment: str, timeout_s: int = 180) -> str:
    return kubectl(f"rollout status deployment/{deployment} --timeout={timeout_s}s",
                   timeout=timeout_s + 30)


def apply_external_db(ip: str, port: int = 5433) -> str:
    """Point the selector-less `shopflow-db` Service at the database's real address."""
    manifest = f"""apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: shopflow-db-1
  namespace: {NAMESPACE}
  labels:
    kubernetes.io/service-name: shopflow-db
addressType: IPv4
ports:
  - name: postgres
    port: {port}
    protocol: TCP
endpoints:
  - addresses: ["{ip}"]
"""
    return kubectl("apply -f -", input=manifest)


def apply_app_secrets() -> str:
    """Create chat-secrets and sqlbot-secrets from .env; DSNs rewritten to `shopflow-db`."""
    s = get_settings()
    out = apply_secret("chat-secrets", {
        "LLM_API_KEY": s.llm_api_key,
        "SESSIONS_DATABASE_URL": sessions_dsn(host="shopflow-db"),
    })
    out += "\n" + apply_secret("sqlbot-secrets", {
        "LLM_API_KEY": s.llm_api_key,
        "DATABASE_URL_RO": with_host(s.database_url_ro, "shopflow-db"),
    })
    return out


IMAGES = ("shopflow-chat:1.0.0", "shopflow-sqlbot:1.0.0", "shopflow-fakellm:1.0.0")
# Third-party images pulled once on the host and side-loaded, so re-creating the cluster doesn't
# download them again from inside every node.
THIRD_PARTY = ("registry.k8s.io/metrics-server/metrics-server:v0.9.1",)


def cluster_exists() -> bool:
    return CLUSTER in sh("kind get clusters", check=False).split()


def create_cluster() -> str:
    """Create the kind cluster (if missing), side-load our images, install metrics-server."""
    log = []
    if not cluster_exists():
        log.append(sh(f"kind create cluster --name {CLUSTER} "
                      f"--config {K8S / 'kind-config.yaml'} --wait 120s", timeout=600))
    for image in THIRD_PARTY:
        if not sh(f"docker images -q {image}"):
            sh(f"docker pull -q {image}", timeout=1200)
    log.append(sh(f"kind load docker-image --name {CLUSTER} {' '.join(IMAGES + THIRD_PARTY)}",
                  timeout=900))
    log.append(kubectl_cluster(f"apply -k {K8S / 'metrics-server'}"))
    return "\n".join(log)


def delete_cluster() -> str:
    return sh(f"kind delete cluster --name {CLUSTER}", timeout=300)


def wait_http(url: str, timeout_s: float = 60) -> float:
    """Poll `url` until it returns 200; return the seconds waited."""
    import httpx

    started = time.perf_counter()
    while True:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return time.perf_counter() - started
        except httpx.HTTPError:
            pass
        if time.perf_counter() - started > timeout_s:
            raise TimeoutError(f"{url} not ready after {timeout_s}s")
        time.sleep(0.2)
