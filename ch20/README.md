# Chapter 20: Deploying to production

> **Tested on:** 2026-11-08 · Ubuntu 22.04 (NVIDIA RTX 3090, 24 GB; GPU not used) · Python 3.12 ·
> openai/gpt-6-luna via OpenRouter · Docker 29.1.3, Compose 5.7.0, kind 0.33.0 (Kubernetes 1.37.0),
> kubectl 1.37.2, Helm 4.3.1 · fastapi 0.162, uvicorn 0.54, sse-starlette 3.6, sqlglot 30.30, uv 0.13.7 ·
> CI/CD workflow and GPU serving manifests: **not executed** (see [Serving it](#serving-it))

On December 11, 2024, between 3:16 PM and 7:38 PM PST, every OpenAI service (ChatGPT, the API,
Sora) was degraded or down. The cause wasn't a model. A new telemetry service rolled out across
the fleet "unintentionally overwhelmed the Kubernetes control plane", DNS depended on that
control plane, and node-level DNS caches hid the damage long enough for the rollout to continue
([OpenAI status](https://status.openai.com/incidents/01JMYB483C404VMPCW726E8MET)). OpenAI's fix
was about *process*: all infrastructure configuration changes now go through phased rollouts.

LLM applications fail in deployment the same way every service does, plus one new way: the
behavior lives in prompts and model versions that change without a code diff. In April 2025 an
update to GPT-4o passed OpenAI's offline evals and small A/B tests and still had to be rolled
back for sycophancy ([OpenAI](https://openai.com/index/expanding-on-sycophancy/)). In September
2025 Anthropic published how three infrastructure bugs, one of them a "routine load balancing
change", degraded Claude's answers for weeks while their evals stayed green
([Anthropic](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues)).

Chapter 19 closed Part V, on models and compute, with a reminder: what you own is the
dataset, the converter and the eval, not the model. This chapter opens Part VI, on scale,
and moves from *which* model we run to *how* we ship the services around it. The eval
idea comes with us: here it becomes a gate that every release has to pass.

We ship two services we built earlier, Chapter 1's support chatbot and Chapter 6's data bot,
the way a backend team ships anything. That means versioned images, Compose for one host and
Kubernetes for many. It also means a CI pipeline with an **eval gate**, and a **prompt canary**
that sends a new system prompt to 10% of customers before everyone gets it. We run all of it on
one machine with kind (Kubernetes in Docker), break the naive version on purpose, and measure
each fix.

The diagram below shows what runs where: the two services, their images, and the cluster they
end up in.

![What runs where](images/20-architecture.svg)

## The problem

Before we deploy anything, let's look at where the two bots stand today. Both work on a laptop.
Chapter 1's chatbot runs as one `uvicorn` process and keeps the conversation history in a
Python dictionary. Chapter 6's data bot answers from a terminal. The platform team now wants
both on the company cluster, behind the same deploy pipeline as every other service. The
support lead also wants to try a shorter, citation-style prompt without betting every customer
on it.

That request hides four questions. This chapter answers each with measurements:

1. **What goes in the image?** Its size, its start-up, and what happens when the cluster has
   no internet.
2. **What breaks when there are three copies instead of one?** Memory, deploys, node
   maintenance.
3. **How do we change a prompt safely?** A percentage canary, and how to decide.
4. **What stops a bad change from shipping at all?** An eval gate in CI.

## Requirements

Those four questions turn into a concrete definition of "done". First, what the deployment must
do:

- **Run anywhere from versioned images.** Both services run from versioned images, on Compose
  and on Kubernetes. Target: ✅ on one host and on a 3-node kind cluster.
- **Remember the conversation on any replica.** A conversation keeps its history even when the
  next turn lands on another copy of the service. 100% of follow-up turns must see the full
  history.
- **Roll out a new prompt gradually.** A new prompt version reaches a configurable share of
  sessions. Each session keeps its version (it is *sticky*), and the change is reversible. The
  real split must stay within ±5 points of the setting, and rollback is one command.
- **Ship nothing that fails the eval gate.** The gate must block a known regression.

Then, how well it must do it:

- **Deploy without dropping anyone.** A rolling deploy under load has 0 failed requests.
- **Keep the image lean and safe.** Under 250 MB, running as a non-root user, and able to start
  with no internet access.
- **Scale with the load.** From 2 up to 6 pods under load, and back to 2 after.
- **Survive node maintenance.** ≥ 1 chat pod keeps serving while nodes are drained.
- **Stay cheap.** Hosted model spend for the whole chapter stays < $1.

At the end of the chapter we check every item against measured numbers.

## The data

To meet those targets we need very little new data, but we do need one place to keep
conversations outside the pods. There is no new business data. The deploy adds one table, for
conversation history, in its own schema. Only a least-privilege role can reach it (`ch20_app`:
`SELECT, INSERT, UPDATE, DELETE` on that table, nothing else):

```mermaid
erDiagram
    chat_sessions {
        text session_id PK
        jsonb messages "user/assistant turns, no system prompt"
        timestamptz updated_at
    }
```

The table holds state; the other input is the evals that will guard each release. The eval gate
reuses two earlier sets, copied into this chapter: Chapter 1's six support
questions with deterministic checks ([`eval/gate_chat.yaml`](01_main-chapter-code/eval/gate_chat.yaml))
and 22 of Chapter 6's data questions with gold SQL ([`eval/gate_sqlbot.yaml`](01_main-chapter-code/eval/gate_sqlbot.yaml)).

To get the database up and run the whole chapter end to end:

```bash
docker compose -f datasets/shopflow/docker-compose.yml up -d   # ShopFlow Postgres on :5433
cd ch20/01_main-chapter-code
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 ch20.ipynb
```

The notebook creates the `ch20` schema and role itself (`ch20_ops.setup_db()`, idempotent). The
notebook also drives the cluster, so we need a few command-line tools first.

### Tools

We need Docker (with Compose v2) plus three static binaries: kind, kubectl and Helm. No root is
needed, because they go in `~/.local/bin`.

```bash
mkdir -p ~/.local/bin
curl -Lo ~/.local/bin/kind https://kind.sigs.k8s.io/dl/v0.33.0/kind-linux-amd64
curl -Lo ~/.local/bin/kubectl https://dl.k8s.io/release/v1.37.2/bin/linux/amd64/kubectl
curl -L https://get.helm.sh/helm-v4.3.1-linux-amd64.tar.gz | tar xz -C /tmp
mv /tmp/linux-amd64/helm ~/.local/bin/ && chmod +x ~/.local/bin/{kind,kubectl,helm}
```

On macOS, `brew install kind kubectl helm` does the same. Check the kubectl download against
`https://dl.k8s.io/release/v1.37.2/bin/linux/amd64/kubectl.sha256`.

## Concepts

With the tools in place, a few ideas make the rest of the chapter easier to follow. We start
with the thing we ship, the image, and work outwards to the cluster and the release process.

### An image is a frozen environment; a multi-stage build keeps only what runs

A **container image** is a file system snapshot plus a start command: Python, our dependencies,
our code. It's the unit we test, ship and roll back, like a compiled binary. A **multi-stage
build** uses one stage (with compilers, `uv`, caches) to *produce* the environment and a second,
clean stage that *copies in* only the result. Build tools never reach production.

Layer order matters as much as stages. Docker caches each instruction, and the first changed one
invalidates everything after it. Dependencies change weekly and code changes hourly, so
dependencies go first.

### Configuration lives outside the image

An image is frozen, but the places it runs are not. The same image must run in CI, staging and
production, so anything that differs between them
(model name, prompt version, canary percentage, database address, API keys) comes from
**environment variables**. Each service reads exactly the variables it needs at start-up
(`services/*/ch20_*/settings.py`) and refuses to start on a bad value. On Kubernetes, plain
settings come from a **ConfigMap** and credentials from a **Secret**.

> [!WARNING]
> A Kubernetes Secret is not encrypted by default. Its values are base64 encoded, which the docs
> call obscuring rather than confidentiality, and they are stored unencrypted in etcd unless you
> enable encryption at rest ([Kubernetes docs](https://kubernetes.io/docs/concepts/configuration/secret/)).
> Restrict who can read Secrets with RBAC, and in production sync them from a real secret store.
> Env vars also show up in `kubectl describe pod` and `docker inspect`.

### Kubernetes in seven objects

Once images and settings are in place, something has to run them on many machines and keep
them running. Kubernetes is a control loop: we declare the state we want in YAML, and
controllers keep nudging the cluster towards it. The table lists the objects this chapter uses;
read the right-hand column if the names are new:

| Object | What it does | Analogy |
|---|---|---|
| **Deployment** | keeps N identical **pods** (containers) running and replaces them gradually on change | a shift manager keeping N cashiers at the tills |
| **Service** | one stable name and address that load-balances over the ready pods | the shop's phone number, whoever answers |
| **ConfigMap** / **Secret** | settings and credentials injected into pods | the briefing sheet and the safe code |
| **Probes** | HTTP checks the kubelet runs on each container: *started?*, *ready for traffic?*, *still alive?* | "is the till open?" vs "is the cashier conscious?" |
| **HorizontalPodAutoscaler** | changes the Deployment's replica count from a metric (CPU here) | calling in extra staff when the queue grows |
| **PodDisruptionBudget** | limits how many pods *voluntary* disruptions (node drains, upgrades) may take at once | "never close every till at the same time" |

**Readiness** and **liveness** sound alike and must check different things. Readiness asks
"should I get traffic right now?" and may depend on the database: a pod whose database is
unreachable should leave the load balancer. Liveness asks "is this process wedged beyond
repair?" and must not depend on anything external, or one database blip restarts every pod at
once. A **startup** probe gives slow starters time before liveness begins. For a 16 GB model
that's minutes, and getting it wrong is a classic:

> [!NOTE]
> **In the wild:** KServe users reported large models "get killed and restarted by the liveness
> probe before they finish loading", and proposed a startup probe allowing 60 × 10 s
> ([kserve#5062](https://github.com/kserve/kserve/issues/5062)). The same bug was filed against
> vLLM's Helm chart in September 2026, where liveness could restart a pod roughly 35 seconds
> after start and end in `CrashLoopBackOff` ([vllm#58544](https://github.com/vllm-project/vllm/issues/58544)).

### What happens when a pod is replaced

Probes tell Kubernetes when a pod can take traffic. The riskiest moment for that is a deploy,
because every deploy replaces pods. Two things then happen *in parallel*, and that's the trap: the
kubelet sends the container `SIGTERM`, and the endpoint controller removes the pod from the
Service. Until every node's kube-proxy has caught up, new connections still arrive at a pod that
is shutting down, and streams that are mid-answer are cut when the process exits.

The diagram traces one old pod through a deploy. Watch the two parallel branches in the middle:

```mermaid
sequenceDiagram
    autonumber
    participant K as kubectl rollout
    participant API as API server
    participant EP as Endpoints / kube-proxy
    participant P as old pod (uvicorn)
    participant C as clients
    K->>API: new pod template
    API->>P: delete pod (grace period starts)
    par
        API->>EP: remove pod from Service
        EP-->>C: new connections go elsewhere (after propagation delay)
    and
        API->>P: preStop hook: sleep 15 s
        Note over P: still serving: in-flight streams finish,<br/>late connections still answered
    end
    API->>P: SIGTERM
    P->>P: uvicorn stops accepting, drains (≤ 25 s)
    API->>P: SIGKILL if still running at 45 s
```

The `preStop` sleep buys time for both: the endpoint removal propagates while the pod still
answers, and in-flight streams finish before `SIGTERM`. The grace period must cover the sleep
plus the drain (`k8s/base/chat.yaml`: 15 + 25 < 45 s). The sleep is a native
`lifecycle.preStop.sleep` action (accepted by our Kubernetes 1.37 cluster), so the image doesn't
need a `sleep` binary or a shell for it.

### Shipping a prompt: canary, and the gate before it

Everything so far is about shipping code safely. But in an LLM service the behavior also lives
in the prompt, and a prompt is code that no compiler checks. So we treat a prompt change like a
risky deploy:

- **Eval gate** (in CI): before an image ships, run the eval set against the *running* services,
  started from the candidate image. Fail the pipeline below a bar.
- **Canary** (in production): send the new version to a small share of real traffic, compare it
  to the stable version on live metrics, then ramp up or roll back.

An eval gate catches what you thought to test. A canary catches what you didn't, at a bounded
blast radius. Anthropic's postmortem names both halves: their evals "simply didn't capture the
degradation users were reporting", and they deploy to small canary groups first.

The flowchart below shows the full path a change takes, from pull request to 100% of traffic.
Each diamond and double box is a point where it can be stopped:

```mermaid
flowchart LR
    PR[pull request] --> T[lint + unit tests]
    T --> B[build images]
    B --> G{eval gate<br/>on Compose stack}
    G -- below bar --> X[blocked]
    G -- pass --> S[staging<br/>canary 0%]
    S --> A1{{approval}} --> C10[canary 10%<br/>30 min, analyze logs]
    C10 -- worse --> R[rollback: set 0]
    C10 -- ok --> A2{{approval}} --> P[50% → 100%]
```

## v0: the naive solution

With the concepts in hand, let's first deploy the bots the way most of us would on a first try,
so we can see what breaks. The first deployment most of us write has two files. The first is a
Dockerfile that copies the repository and `pip install`s whatever is current:

```dockerfile
# ch20/01_main-chapter-code/services/chat/Dockerfile.naive
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn sse-starlette openai tiktoken "psycopg[binary]"
CMD ["uvicorn", "--factory", "ch20_chat.app:create_app", "--host", "0.0.0.0", "--port", "8080"]
```

And a manifest that runs three copies of it behind a Service
([`k8s/v0/chat-naive.yaml`](01_main-chapter-code/k8s/v0/chat-naive.yaml)): Chapter 1's
in-memory sessions, no probes, no resource requests, no disruption budget, no shutdown handling.
`kubectl apply`, three pods go `Running`, `curl` answers. It works.

The chat service itself is Chapter 1's API with three production changes we keep in every
version below. It reads its settings only from the environment and refuses to start on a bad
one. The system prompt is a versioned file, `prompts/support-<version>.md`, selected per request.
It logs one JSON line per request (pod, prompt version, status, TTFT, tokens, cost), which is all
the canary analysis needs later.

## Measuring v0

v0 answers a `curl`, but that tells us little about how it behaves under real traffic and real
deploys. So we measure it. We use a **fake LLM**, a 60-line OpenAI-compatible server
([`services/fakellm`](01_main-chapter-code/services/fakellm/ch20_fakellm.py)) deployed in the
cluster. It streams 40 words in about 2.3 s. It starts every answer with how many user messages
it received, so a client can check whether the service sent the whole conversation. Load tests
against it cost nothing and don't depend on a provider's latency.

We take four measurements, all in the notebook.

**The image.** The naive image is 1,206 MB and runs as root. After a one-line code change it
rebuilt in 69.7 s, because `COPY . .` comes before `pip install` and invalidates it; that
number is mostly download time and swings with the network (230 s in an earlier run on the
same host). Worse, it doesn't start without internet access:

```
                     image                                             result  seconds
0       ch20-scratch:naive  exited (1): requests.exceptions.ConnectionErro...      1.4
1  ch20-scratch:multistage                                            serving      2.0
```

`tiktoken` downloads its tokenizer file the first time it's used, which in our app is at import
time. On a network with no egress (common in production clusters) the naive container crashes
on start; Kubernetes would restart it forever (`CrashLoopBackOff`).

**Memory across replicas.** 30 simulated customers each have a 3-turn conversation; the fake
LLM tells us how many user messages it received:

```
      requests  ok  history_complete  pods_seen
turn
1           30  30                30          3
2           30  30                11          3
3           30  30                 3          3
```

Every request succeeded, and the bot forgot the conversation 19 times out of 30 on the second
turn and 27 out of 30 on the third. With three replicas, a follow-up lands on the pod that
holds the history about a third of the time (and both follow-ups only a ninth). Nothing errors,
no dashboard turns red; customers just get a bot that doesn't listen.

> [!WARNING]
> A browser keeps its connection open, and kube-proxy balances *connections*, not requests. So
> a manual test from one browser tab usually hits the same pod and looks fine. Our load
> generator opens a new connection per request, like many different users (or a proxy that
> recycles connections) would.

**A rolling restart under load.** 20 users stream answers continuously while we run
`kubectl rollout restart deployment/chat`, which is what every deploy does:

```
rollout took 2 s
{'requests': 879, 'ok': 743, 'failed': 136, 'failed_%': 15.5,
 'failure_kinds': {'connect_error': 104, 'incomplete': 32}}

  9.4s  incomplete     RemoteProtocolError: peer closed connection without sending complete message body (incomplete chunked read)
 10.8s  connect_error  ReadError:
```

15.5% of requests failed during a routine deploy. The 32 `incomplete` ones are worse than
errors: the customer saw half an answer stop mid-sentence. And `kubectl rollout status`
declared success after 2 s, because without a readiness probe a pod counts as available the
moment its process starts, before uvicorn even listens.

**Placement.** The pod listing after `kubectl apply` shows two of the three chat pods on the
same worker (`ch20-worker`). Nothing in v0 asks the scheduler to spread them, so one node
failure can take most of the capacity with it.

## Iterating

Each of the four measurements found a real problem. We fix them one at a time, starting with
the image, because every later fix ships inside it.

### 1. A multi-stage image that starts offline

What's wrong: the naive image is 1.2 GB and runs as root. It reinstalls every dependency on
each code change, floats on unpinned versions and can't start without the internet.

Options: slim base + `pip install -r requirements.txt` (simple, unpinned transitive deps);
multi-stage with `uv sync --locked` (pinned, small); distroless/Chainguard bases (smaller still,
no shell for debugging).

> [!IMPORTANT]
> **Decision: multi-stage `python:3.12-slim` + `uv sync --locked` over a single-stage or a
> distroless image.** The lock file pins every transitive dependency, so the image that passed
> the eval gate is the image that ships. The builder stage keeps `uv` and caches out of
> production, and the slim runtime keeps a shell for `kubectl exec` while we're learning.
> Choose a distroless base when your security team scans images and every package is a
> finding; you lose the shell.

```dockerfile
# ch20/01_main-chapter-code/services/chat/Dockerfile (excerpt)
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 UV_NO_DEV=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-cache
# tiktoken downloads its encoding file on first use. Bake it in so pods start offline.
ENV TIKTOKEN_CACHE_DIR=/app/tiktoken
RUN for attempt in 1 2 3 4 5; do \
      /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base')" && break; \
      sleep 5; \
    done && test -n "$(ls /app/tiktoken)"

FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/tiktoken /app/tiktoken
COPY ch20_chat ./ch20_chat
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 TIKTOKEN_CACHE_DIR=/app/tiktoken
USER 10001
```

Measured on the same scratch copy of the service:

| Dockerfile | size | layers | user | rebuild after a one-line change | starts with no internet |
|---|---|---|---|---|---|
| naive | 1,206 MB | 10 | root | 69.7 s | no: exits in 1.4 s (`ConnectionError`) |
| multi-stage | 191 MB | 9 | 10001 | 0.2 s | yes: serving in 2.0 s |

![Image size and rebuild time](images/20-images.png)

The data bot's image is built the same way. All three images (chat, sqlbot, fake LLM) are
tagged `1.0.0`, never `latest`.

> [!WARNING]
> **`:latest` and kind don't mix.** `kind load docker-image` copies a local image into the
> nodes, but for an image tagged `:latest` (or untagged) Kubernetes defaults to
> `imagePullPolicy: Always`, tries a registry, and fails with `ErrImagePull` even though the image
> is right there ([kind docs](https://kind.sigs.k8s.io/docs/user/quick-start/),
> [iximiuz](https://iximiuz.com/en/posts/kubernetes-kind-load-docker-image/), who "wasted a day"
> on it). Pin a tag and set `imagePullPolicy: IfNotPresent`; `tests/test_ch20_manifests.py`
> fails the build if any Deployment doesn't.

> [!TIP]
> No BuildKit? This host's Docker has no `buildx` plugin, so `RUN --mount=type=cache` (the uv
> docs' preferred pattern) fails with "the --mount option requires BuildKit". The Dockerfiles
> use a plain `COPY pyproject.toml uv.lock` layer instead, which caches just as well for code
> changes; the cache mount only speeds up rebuilds after the lock file changes.

### 2. Sessions in Postgres, not in the pod

The image is now small and starts offline, but it still keeps conversations in memory. What's
wrong: the bot forgot 19 of 30 conversations on the second turn, because each replica keeps its
own history in memory.

Options: sticky sessions at the load balancer (session affinity: cheap, but a restart or a
scale-down still loses the history, and load spreads unevenly); Redis (fast, one more stateful
service to run); Postgres (already there, durable, a few milliseconds per turn).

> [!IMPORTANT]
> **Decision: Postgres over Redis or sticky sessions.** ShopFlow already runs Postgres, a chat
> turn takes seconds, and two small queries per turn are noise next to the model call. Sticky
> sessions only hide the bug until the next deploy. Pick Redis when you need sub-millisecond
> reads or TTL-based expiry at high volume (Chapter 15 runs it as a semantic cache).

```python
# ch20/01_main-chapter-code/services/chat/ch20_chat/sessions.py (excerpt)
class PostgresSessionStore:
    """One short connection per operation: simple and stateless. A pool is the next step."""

    def get(self, session_id: str) -> list[Message]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT messages FROM ch20.chat_sessions WHERE session_id = %s", (session_id,)
            ).fetchone()
        return list(row[0]) if row else []

    def put(self, session_id: str, turns: list[Message]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ch20.chat_sessions (session_id, messages) VALUES (%s, %s) "
                "ON CONFLICT (session_id) DO UPDATE "
                "SET messages = EXCLUDED.messages, updated_at = now()",
                (session_id, Jsonb(turns)),
            )
```

The stored history has no system prompt. Chapter 1 stored it as the first message; with
versioned prompts that would freeze each conversation on whatever prompt it started with, and a
rollback would never reach it. The prompt is chosen per request and prepended.

The database lives outside the cluster. A **Service without a selector** plus an EndpointSlice
gives it a stable in-cluster name (`shopflow-db:5433`), so the DSN in the Secret never contains
an IP address ([`k8s/base/external-db.yaml`](01_main-chapter-code/k8s/base/external-db.yaml)).

Same 30 conversations on the hardened deployment:

```
      requests  ok  history_complete  pods_seen
turn
1           30  30                30          2
2           30  30                30          2
3           30  30                30          2
```

90 of 90 turns saw the full history, served by both pods. The cost is two short Postgres
connections per turn; a connection pool is the obvious next step at higher traffic.

### 3. Probes and graceful shutdown

Conversations now survive any replica, so the next weakness is the deploy itself. What's wrong:
136 of 879 requests (15.5%) failed during one rolling restart, 32 of them cut mid-answer, and
the rollout reported success before the new pods could serve.

The fix is four settings in [`k8s/base/chat.yaml`](01_main-chapter-code/k8s/base/chat.yaml):

```yaml
# ch20/01_main-chapter-code/k8s/base/chat.yaml (excerpt)
    spec:
      terminationGracePeriodSeconds: 45
      containers:
        - name: chat
          startupProbe:            # up to 60 s to import and load prompts before liveness starts
            httpGet: {path: /healthz, port: http}
            periodSeconds: 2
            failureThreshold: 30
          readinessProbe:          # dependencies (session DB) checked here, not in liveness
            httpGet: {path: /readyz, port: http}
            periodSeconds: 5
            failureThreshold: 2
          livenessProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 10
            failureThreshold: 3
          lifecycle:
            preStop:
              sleep: {seconds: 15}
```

plus `maxUnavailable: 0, maxSurge: 1` in the rollout strategy, so capacity never drops below
the current replica count while pods are replaced.

The probe *handlers* matter as much as the probe settings. Chapter 1's `/healthz` was a plain
`def`, so FastAPI ran it in the same 40-thread pool as the streaming `/chat` handlers. Under load
it queued behind them: Chapter 21's load test measured it taking seconds, longer than our 2 s
probe timeout, which would let liveness restart perfectly healthy pods exactly when they're
busiest. Both probes are now `async def` and run on the event loop; `/readyz` uses psycopg's
async connection for its database check
([`app.py`](01_main-chapter-code/services/chat/ch20_chat/app.py)), and a unit test pins it.

> [!IMPORTANT]
> **Decision: a `preStop` sleep over application-level drain logic.** We could make the app
> catch `SIGTERM`, flip `/readyz` to 503 and wait for streams to finish. But the endpoint
> removal is already happening in parallel; the app can't speed it up, only wait for it. A
> fixed 15 s sleep covers propagation plus our longest normal stream with zero code. Write
> drain logic when streams can last minutes (agents, long generations): then the grace period
> has to grow, and the app should stop accepting new work explicitly.

The same rolling restart under the same load (20 concurrent streams for 90 s):

```
rollout took 7 s
{'requests': 760, 'ok': 760, 'failed': 0, 'failed_%': 0.0, 'failure_kinds': {}}
```

![Requests completed and failed per 2 s during a rolling restart](images/20-rollout.png)

Zero failures out of 760. The rollout now takes 7 s instead of 2 s, because "done" means the new
pods passed their readiness probe; the old pods then keep serving for their 15 s `preStop`
before they get `SIGTERM`. (The hardened run completes fewer requests in total, 760 vs 879,
only because v0's failures returned instantly and freed their workers for another request.)

> [!NOTE]
> **In the wild:** Rakuten's engineers hit the same race on their Kubernetes services and fixed
> it the same way: a `preStop` sleep of 20 s with a 60 s termination grace period, noting that
> endpoint removal doesn't touch connections that are already established
> ([Rakuten Engineering](https://engineering.rakuten.today/post/graceful-k8s-delpoyments/)). A
> Kubernetes maintainer explains the same behavior in
> [kubernetes#105703](https://github.com/kubernetes/kubernetes/issues/105703).

The other half of the probe story is what happens when a dependency fails. We pointed the
`shopflow-db` Service at an unroutable address for 25 s:

```
during the outage:
POD                    READY   RESTARTS
chat-5cd8484bc-c2f9h   false   0
chat-5cd8484bc-krzzf   false   0
...                    (all 6 pods: false, 0 restarts)
request during outage: ReadError

DB back; Service serving again after 3.5 s
```

Readiness took every pod out of the Service, and liveness restarted none of them (the HPA had
scaled to 6 during the rollout test). With the database check in the liveness probe instead,
the same 25 s blip would have restarted all six pods, and they would have come back to a
database that was still down. The outage is still an outage, since a chatbot with no session
store has nothing to serve, but recovery takes seconds, not a restart storm.

### 4. Autoscaling and disruption budgets

Deploys no longer drop requests, but the number of pods is still fixed by hand. What's wrong: v0
has a fixed replica count. Three pods idle at night and saturate at peak, and a node upgrade
evicts every pod on the node at once.

The HPA targets 60% of the chat container's 100m CPU request, between 2 and 6 pods
([`k8s/base/chat-scaling.yaml`](01_main-chapter-code/k8s/base/chat-scaling.yaml)). We shortened
the scale-down stabilization window from the default 300 s to 60 s so the experiment fits in a
notebook; keep the default in production.

> [!IMPORTANT]
> **Decision: CPU-based HPA for the chat proxy, not for model servers.** Our pods do
> per-request work (tokenizing the 1,150-token prompt, JSON-encoding every SSE event, two
> Postgres round trips), so CPU rises with concurrent streams and the built-in metric works with
> nothing but metrics-server. GPU model servers are different: GPU utilization saturates early
> and CPU says nothing, so scale them on queue depth (vLLM's `num_requests_waiting`, via KEDA or
> the Prometheus adapter), as Google's GKE guidance and the production-stack chart do.

40 concurrent streams for 150 s, then nothing, starting from 2 pods:

| seconds | replicas | avg CPU (% of request) | event |
|---|---|---|---|
| 0 | 2 | 5 | load starts |
| 30 | 2 | 425 | metrics-server reports the load (about 30 s lag) |
| 45 | 4 | 370 | first scale-up (policy: +2 pods per 15 s) |
| 61 | 6 | 210 | at `maxReplicas` |
| 150 | 6 | ~180 | load stops |
| 197 | 6 | 3 | CPU back to idle in the metrics |
| 243 → 329 | 5 → 2 | 3–4 | one pod per 30 s after the 60 s window |

![HPA replicas and CPU during and after load](images/20-hpa.png)

2,534 of 2,534 requests succeeded, including through four scale-downs (the `preStop` sleep at
work again), and p95 latency stayed between 2.40 and 2.51 s in every 30 s window. Two things
to notice. The HPA reacts to what metrics-server saw 15–30 s ago, so it can't absorb a spike
that lasts seconds; keep headroom in `minReplicas`. And at the 6-pod ceiling CPU still sat at
140–180% of the 100m request without hurting latency: the request is set lower than what a
busy pod really uses, which makes a 60% target aggressive. Calibrate the request from a load
test (Chapter 21) before you trust the target.

> [!NOTE]
> **In the wild:** Google's GKE best-practice guide for LLM inference warns that "using CPU
> metrics alone for autoscaling can lead to suboptimal performance and costs" and recommends
> queue size as the first signal ([Google Cloud](https://cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling)).
> The vLLM production-stack chart ships a KEDA trigger on `vllm:num_requests_waiting` for the
> same reason.

A node drain is the other way pods disappear. We drained both workers in turn while 4 users
kept chatting. The chat PDB says `minAvailable: 1`; the fake LLM got the same budget so it
wouldn't vanish either. Draining the first worker evicted its pods one by one; draining the
second, with nowhere left to schedule replacements (the control-plane node is tainted), hit the
budgets:

```
pod/chat-5cd8484bc-jszvh evicted
pod/sqlbot-7678b865d7-xj9tw evicted
error when evicting pods/"chat-5cd8484bc-xb5ww" -n "shopflow" (will retry after 5s): Cannot evict pod as it would violate the pod's disruption budget.
error when evicting pods/"sqlbot-7678b865d7-7n2wl" -n "shopflow" (will retry after 5s): Cannot evict pod as it would violate the pod's disruption budget.
error: unable to drain node "ch20-worker2" due to error: [... global timeout reached: 45s ...]

NAME                   READY   STATUS    RESTARTS   AGE
chat-5cd8484bc-8569z   0/1     Pending   0          46s
chat-5cd8484bc-xb5ww   1/1     Running   0          11m

requests during both drains: {'requests': 252, 'ok': 252, 'failed': 0, 'failed_%': 0.0}
```

The drain refused to take the last chat pod, and all 252 requests during both drains succeeded.
That refusal is the point: a cluster upgrade blocks and pages a human instead of taking the
service down. A PDB can also block forever: `minAvailable` equal to the replica count makes every
drain hang, so a unit test checks `minAvailable` is below the HPA's `minReplicas`.

> [!WARNING]
> PDBs only limit *voluntary* disruptions through the eviction API (drains, cluster
> autoscaler). They don't apply to a Deployment's own rolling update, which is governed by
> `maxUnavailable` ([Kubernetes docs](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/)),
> and they can't stop a node from crashing.

### 5. An eval gate in front of every image

The service now survives deploys, load and node drains. What's wrong: nothing above checks
whether the service still *answers well*. A prompt edit, a model bump or a "harmless" cleanup of
the semantic context ships as easily as a typo fix.

The gate ([`ch20_eval_gate.py`](01_main-chapter-code/ch20_eval_gate.py)) runs against the
services *as deployed*, the Compose `prod` profile started from the candidate images, so it
tests image, configuration and prompts together. The chat checks are deterministic substring
rules on Chapter 1's six questions; the data checks are Chapter 6's execution accuracy on 19
answerable questions plus "no write or PII query ran" on 3 adversarial ones. Exit code 0 or 1
is all CI needs.

```
$ docker compose --profile prod up -d --wait          # in the notebook: healthy after 5.8 s
$ uv run python ch20_eval_gate.py --chat-url http://localhost:8120
exit code 0 {"passed": true, "failures": [], "chat_pass_rate": 1.0, ...}
PASS return-window            You can return most items within 30 days of delivery. Items must be unused and in their or
PASS failed-payment           A failed payment isn’t charged, but your bank may show a temporary authorization hold that
PASS paypal-indonesian        Ya, ShopFlow menerima pembayaran melalui PayPal.
PASS bereavement-not-in-faq   I’m sorry for your loss. The FAQ doesn’t say whether ShopFlow offers bereavement discounts
PASS out-of-scope             Sorry, I can only help with ShopFlow orders, shipping, returns, payments, and accounts.
PASS injection                I can’t approve or issue refunds or look up order 1234. To start a return, go to **Account
```

To see whether the gate earns its place, we sent it two pull requests that "trim the planner
prompt" of the data bot to save tokens. PR 1 deletes the `conventions` block and the worked
examples from `semantic_context.yaml`. PR 2 also deletes the glossary. Each PR's image is
built and gated exactly like main.

Our first gate design was the obvious one: ask each question once, pass at 90%. But one flaky
answer moves a 19-question score by 5.3 points, and the bot is flaky on some questions even on
main (below, `s8` fails in 1 of 3 repeats on the unchanged image). A 90% bar on one pass is
mostly measuring luck. We changed two things:

- **Repeats.** Each data question is asked 3 times (57 answers per run, about $0.013 on
  gpt-6-luna), so one unlucky answer moves the score by 1.8 points.
- **An item-level comparison with main** (`--baseline`). A question that main gets right 3 times
  out of 3 and the candidate gets wrong 3 times out of 3 blocks the merge, whatever the average.
  Three-for-three both ways is hard to explain by sampling noise.

```python
# ch20/01_main-chapter-code/ch20_eval_gate.py (excerpt)
def flipped_items(baseline: list[dict], candidate: list[dict]) -> list[str]:
    """Items that passed every repeat on the baseline and failed every repeat on the candidate."""
    base, cand = counts(baseline), counts(candidate)
    return sorted(i for i, (p, n) in cand.items()
                  if p == 0 and i in base and base[i][0] == base[i][1])
```

The run in the notebook:

| change | planner prompt tokens | execution accuracy (57 answers) | adversarial safe | items failed (of 3 repeats) | flipped | gate |
|---|---|---|---|---|---|---|
| main | 2,011 | 93.0% | 100% | j1 ×3, s8 ×1 | – | pass |
| PR 1: drop conventions + examples | 1,071 | 94.7% | 100% | j1 ×3 | – | pass |
| PR 2: … and the glossary | 916 | 86.0% | 100% | s1 ×3, j1 ×2, s8 ×2, t3 ×1 | s1 | **BLOCKED** |

PR 1 cuts the planner prompt by 47% and scores within noise of main, so the gate lets it
through: evidence that the examples and conventions had become redundant with the column
descriptions and the glossary. That's the less obvious value of a gate: it makes cheap changes
*safe to ship*, not only bad changes impossible.

PR 2 fails both rules. The flipped item is `s1`, Chapter 6's headline question:

```
[main, 3 of 3]  There are 100 transactions (orders) today, ...
[PR 2, 3 of 3]  There are 102 transactions (payment attempts, including succeeded and failed) today, ...
                SELECT COUNT(*) AS transaction_count FROM payments WHERE ...
```

Without the glossary line that defines a transaction as an order, the model counts payment
attempts: the exact ambiguity Chapter 6 was built around, back after a cleanup that looked
harmless in review.

`j1` (revenue by category) fails on main too, every time: the model allocates order totals to
categories proportionally, while our gold SQL attributes line-item value. Both are defensible
business definitions. It's a known, stable failure, which is precisely why the flip rule
ignores items that already fail on main.

> [!IMPORTANT]
> **Decision: repeats plus an item-level baseline over a single higher bar.** Raising the bar
> to 95% on one pass would have blocked main itself half the time. Repeats shrink the noise,
> and the flip rule catches a regression concentrated on one important question that an average
> over 19 hides. The cost is 3× the tokens (about $0.013 per gate run here) and a baseline file
> to refresh on every merge to main (`eval/results/gate-sqlbot-baseline.json`). Use a bigger
> eval set instead when a run costs dollars rather than cents, or when you can't keep a
> baseline current.

> [!NOTE]
> **In the wild:** GitHub says every new model for Copilot's next edit suggestions goes through
> "offline testing, internal dogfooding, and online A/B experiments", the last on a small
> percentage of real requests while tracking acceptance, hide and latency metrics
> ([GitHub Blog](https://github.blog/ai-and-ml/github-copilot/evolving-github-copilots-next-edit-suggestions-through-custom-model-training/)).
> Offline evals gate; live experiments decide.

### 6. A prompt canary

The gate catches what we thought to test. Some changes need real customers to judge. What's
wrong: the support lead wants prompt v2
([`support-v2.md`](01_main-chapter-code/services/chat/ch20_chat/prompts/support-v2.md)): at
most 2 sentences, and a `[FAQ section]` citation at the end of every answer that comes from the
FAQ. The gate's six checks pass on it, but six questions say nothing about tone, length or how
real customers phrase things. Shipping it to everyone is a bet.

Options: **blue/green** (run both versions, flip all traffic at once: instant rollback, but every
customer is exposed at the flip); **shadow** (send real traffic to v2 too and discard its
answers: zero customer risk, double the model cost, and no customer reaction to measure);
**canary** (a small, growing share of real sessions on v2).

> [!IMPORTANT]
> **Decision: an in-app percentage router over a service-mesh canary.** Istio, Linkerd or Argo
> Rollouts split traffic between two *Deployments* by request. A prompt version is a config
> value, not a Deployment, and a chat conversation must keep its version from turn to turn, so
> the split has to be by session. Ten lines in the app do that: hash the session id (salted
> with the canary's name, so each canary picks a different slice of customers) into 10,000
> buckets and compare with `CANARY_PERCENT`. Use a mesh when the canary is a new image or a new
> model server; you can combine both.

```python
# ch20/01_main-chapter-code/services/chat/ch20_chat/canary.py
def bucket(session_id: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{session_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % BUCKETS


def choose_version(session_id: str, stable: str, canary: str | None, percent: float) -> str:
    """Return the prompt version for this session: `canary` for `percent`% of sessions."""
    if not canary or percent <= 0:
        return stable
    if percent >= 100:
        return canary
    return canary if bucket(session_id, canary) < percent * BUCKETS / 100 else stable
```

Two properties follow, and both are unit-tested on 20,000 session ids: a session keeps its
version while the percentage is unchanged, and raising the percentage only ever moves sessions
from v1 to v2 (setting it to 0 moves everyone back).

The percentage lives in the kustomize-generated ConfigMap. Its name carries a hash of its
content, so changing `CANARY_PERCENT` changes the pod template and triggers a normal rolling
update, which we already know drops nothing. `ch20_canary.py set 10` writes a two-line overlay
and applies it; `ch20_canary.py analyze` reads the pods' JSON logs and compares versions.

The run, on `openai/gpt-6-luna` (as of 2026-11: $0.10 / $0.50 per 1M input/output tokens on
OpenRouter), with the same 120 customers asking one question in each phase:

```
rolled out CANARY_PERCENT=10 in 10 s
client side: {'v1': 109, 'v2': 11} | answered by pods of the previous revision (still draining): 2
verdict: ('wait', ['canary has 11 requests, need 20'])

rolled out CANARY_PERCENT=50 in 10 s
verdict: ('promote', ['within thresholds'])
customers on v2: 11 at 10%, 58 at 50%; still on v2 after the ramp: 11/11
```

Three things in those lines are worth a pause. At 10%, 120 sessions give the canary 11
requests, too few to judge, and the analysis says so instead of guessing. All 11 customers who
had v2 at 10% still had it at 50%: nobody flip-flops between prompts during a ramp. And 2
requests landed on pods of the *previous* revision while they drained: the `preStop` window
working as designed, which also means a canary's first seconds are slightly mixed.

The comparison over both phases (240 real answers):

| version | answers | errors | words / answer | completion tokens | answers citing a section | TTFT p50 | total p95 | $ / 1k answers |
|---|---|---|---|---|---|---|---|---|
| v1 (stable) | 171 | 0 | 17.4 | 26.9 | 0% | 0.998 s | 1.807 s | 0.033 |
| v2 (canary) | 69 | 0 | 16.5 | 26.9 | 100% | 0.814 s | 1.504 s | 0.039 |

![Canary comparison](images/20-canary.png)

```
[v1] Who pays for return shipping?
  -> Return shipping is free for defective or wrong items. Otherwise, $5.99 is deducted from your refund.

[v2] Who pays for return shipping?
  -> Return shipping is free for defective or wrong items; otherwise, $5.99 is deducted from your refund. [Returns and refunds]
```

v2 did what it was written for: every answer cites its FAQ section, at the same length and the
same output tokens. It wasn't slower. Its cost per 1,000 answers came out 18% higher, with
identical output tokens, so the difference is on the input side: v2's longer system prompt, and
fewer v2 requests to share each prompt-cache write ([INFERENCE]: we didn't break the cost down by
cached tokens). We promoted it, then practised the way back:

```
promoted to 100% in 10 s
deployment.apps/chat rolled back
Warning: resource deployments/chat was previously managed with 'kubectl apply'. Rolling back will
not update the kubectl.kubernetes.io/last-applied-configuration annotation, ...
undo took 9 s -> {'status': 'ready', ..., 'canary_percent': 100.0}
```

Two surprises. `kubectl rollout undo` works (it restores the previous ReplicaSet, which points
at the previous hash-named ConfigMap), but kubectl itself warns that the next `kubectl apply`
may be confused: the cluster now disagrees with what was last applied from git. And the
`/readyz` call right after the undo still reported 100%. We didn't record which pod answered it;
the likeliest explanation is an old pod in its 15 s `preStop` window, the same effect as the 2
"previous revision" requests above [INFERENCE]. In a pipeline, roll back the way you rolled
forward (`ch20_canary.py set 0`), so git stays the source of truth, and verify a rollback from
the logs of the new pods, not from one request.

> [!NOTE]
> **In the wild:** In April 2025 OpenAI rolled back a GPT-4o update that had made ChatGPT
> sycophantic. Its offline evaluations "generally looked good" and the A/B tests suggested the
> small number of users who tried it liked it; OpenAI concluded it must "treat model behavior
> issues as launch-blocking" and that "there's no such thing as a 'small' launch"
> ([OpenAI](https://openai.com/index/expanding-on-sycophancy/)). Canaries measure what you chose
> to measure; tone and safety need their own canary metrics (Chapters 13 and 14).

## Production hardening

The six fixes above each answered a measured failure. The hardened manifests and images also do
several things we didn't stage a failure for, but that a real cluster expects:

- **Least privilege, per service.** The data bot gets only the read-only DSN; the chat service
  gets a role that can touch one table (`ch20.chat_sessions`). Each service declares its own
  settings instead of using the book's shared `llmprod.get_settings()`, which demands every
  credential the book uses, including the admin DSN.
- **Fail fast on config.** An unknown prompt version, a percentage outside 0–100 or a missing
  key raises at start-up; the pod never becomes ready, and the rollout stops with the old pods
  still serving (`maxUnavailable: 0`).
- **Containers.** Non-root (uid 10001), read-only root filesystem with an `emptyDir` on `/tmp`,
  all capabilities dropped, `RuntimeDefault` seccomp, no privilege escalation. The Compose `prod`
  profile does the same with `read_only`, `cap_drop` and `no-new-privileges`.
- **Resources.** CPU and memory requests (the HPA needs the CPU request), memory limits, and no
  CPU limit, so a busy pod is throttled by its neighbors' requests, not by an arbitrary cap.
- **Placement.** `topologySpreadConstraints` on the hostname keeps replicas on different nodes.
- **Bounded work.** `max_tokens` on every call, a 30 s provider timeout with one retry, 4,000
  characters per message, an 8,000-token history budget (all from Chapter 1).
- **One log line per request**, JSON, with pod, prompt version, status, TTFT, tokens and cost.
  It's what `ch20_canary.py analyze` reads, and what Chapter 11 ships to Loki.
- **Static checks** in `tests/test_ch20_manifests.py`: probes on every container and liveness
  never on `/readyz`, pinned tags, non-root, `preStop` shorter than the grace period, PDBs that
  can't block every drain, Service selectors and named ports that match, kind port mappings that
  match the NodePorts.

> [!WARNING]
> **Streaming through an ingress.** We expose NodePorts, which pass SSE straight through. Put an
> ingress controller in front and two things change: buffering (Chapter 1's
> `X-Accel-Buffering: no` header still matters for plain nginx) and reloads. One ingress-nginx
> user found long SSE streams cut a fixed time after *any* Ingress change, because each change
> reloads nginx and old workers are killed after `worker-shutdown-timeout`
> ([ingress-nginx#14090](https://github.com/kubernetes/ingress-nginx/issues/14090)). And
> ingress-nginx itself was retired in March 2026; the Kubernetes project points new setups to
> Gateway API ([Kubernetes blog](https://kubernetes.io/blog/2025/11/11/ingress-nginx-retirement/)).

## Serving it

With the hardened images and manifests ready, here is how to run them yourself, from one host
up to a full pipeline.

### On one host: Compose

Compose is the quickest way to run both services together on a single machine:

```bash
cd ch20/01_main-chapter-code
docker build -t shopflow-chat:1.0.0 services/chat
docker build -t shopflow-sqlbot:1.0.0 services/sqlbot
export LLM_API_KEY=...                                  # from your secret store
export CH20_SESSIONS_DATABASE_URL=postgresql://ch20_app:...@host.docker.internal:5433/shopflow
export CH20_DATABASE_URL_RO=postgresql://shopflow_ro:...@host.docker.internal:5433/shopflow
docker compose --profile prod up -d --wait              # chat on :8120, data bot on :8020
docker compose --profile dev up                         # source mounted, --reload, fake LLM
```

```bash
curl -s -X POST localhost:8020/ask -H 'content-type: application/json' \
  -d '{"question": "How many transactions are there today?"}'
```

`/ask` returns the answer, the SQL that ran, up to 50 rows, the true row count, attempts,
latency, usage and the pod name. The chat API is Chapter 1's (`POST /chat`, SSE), plus
`X-Prompt-Version` and `X-Served-By` headers and the same two fields in the final `done` event.

> [!TIP]
> The compose file uses `${LLM_API_KEY:?...}` so a missing key fails loudly. The catch: even
> `docker compose down` then refuses to run without the variables set.

### On Kubernetes: kind

Compose covers one host. To see replicas, probes and the canary at work, we need a cluster, and
kind gives us one inside Docker:

```bash
kind create cluster --name ch20 --config ch20/01_main-chapter-code/k8s/kind-config.yaml
kind load docker-image --name ch20 shopflow-chat:1.0.0 shopflow-sqlbot:1.0.0 shopflow-fakellm:1.0.0
kubectl apply -k ch20/01_main-chapter-code/k8s/metrics-server
# Secrets and the database endpoint come from .env and the host; the notebook does this with
# ch20_ops.apply_app_secrets() and ch20_ops.apply_external_db(ch20_ops.kind_gateway()).
kubectl apply -k ch20/01_main-chapter-code/k8s/base           # real model
kubectl apply -k ch20/01_main-chapter-code/k8s/overlays/fakellm   # or: fake LLM for load tests
uv run python ch20/01_main-chapter-code/ch20_canary.py set 10
uv run python ch20/01_main-chapter-code/ch20_canary.py analyze --since 15m
```

The chat UI from Chapter 1 is served at http://localhost:8320 and the data bot at
http://localhost:8420 (kind maps the NodePorts to the host).

### The CI/CD pipeline

So far we have run each step by hand. In a team, a pipeline runs them in order on every change.
[`github-workflows/deploy.yml`](01_main-chapter-code/github-workflows/deploy.yml) wires the
pieces together. It lives in the chapter folder, not in this repository's `.github/`, because it
deploys to clusters we don't have. Copy it into your own repository's `.github/workflows/`. The
table shows each job and what stops it:

| Job | What it does | Gate |
|---|---|---|
| `test` | `ruff` + `pytest ch20` (no network, no model) | fails on any test |
| `build-and-gate` | seeds a Postgres service container, builds both images, starts the Compose `prod` profile, runs `ch20_eval_gate.py`, uploads `gate.json`, pushes the images tagged with the commit SHA | eval gate exit code |
| `staging` | `ch20_canary.py set 0 --registry … --tag <sha>`: deploy the gated images with the canary off | `environment: staging` |
| `canary` | `set 10`, wait 30 min, `analyze`; on a bad verdict, `set 0` and fail | `environment: production-canary` (required reviewers) |
| `promote` | `set 50`, then `set 100` | `environment: production` (second approval) |

Two details that cost us a broken deploy on paper before we caught them:

- Every `set` passes `--registry` and `--tag`. Without them the generated overlay renders the
  base manifests' image names, and changing the canary percentage would quietly roll production
  back to whatever image the base names.
- The cluster Secrets are not created by the pipeline. In production they're synced from a
  secret manager; the workflow never sees the LLM key that the pods use.

> [!CAUTION]
> **Not executed by the author** — requires a GitHub repository with Actions, a container
> registry and two clusters. Written against the GitHub Actions docs for
> [service containers](https://docs.github.com/en/actions/tutorials/use-containerized-services/use-docker-service-containers),
> [environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)
> and [concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)
> (accessed 2026-11-07) and checked with `actionlint` 1.7.12 (no findings). Every step it runs
> (`pytest`, `docker build`, `docker compose --profile prod up --wait`, the gate, `ch20_canary.py
> set/analyze`) was executed locally in this chapter.

### Self-hosting the model on GPUs

Everything above calls a hosted model. When volume, privacy or cost push you to self-host
(Chapters 16–18), the model server becomes one more Deployment, with three differences that
matter for deploys: pods need a GPU (`nvidia.com/gpu: 1` and the NVIDIA runtime), start-up takes
minutes (weights download and load), and CPU is the wrong autoscaling signal. Two common ways to
run vLLM on Kubernetes, both written for one 24 GB GPU serving `Qwen/Qwen3-8B`:

> [!CAUTION]
> **Not executed by the author** — requires a Kubernetes node with an NVIDIA GPU. GPUs on kind
> need `nvkind` and `accept-nvidia-visible-devices-as-volume-mounts = true` in
> `/etc/nvidia-container-runtime/config.toml` ([NVIDIA/nvkind](https://github.com/NVIDIA/nvkind));
> on the book's host that setting is off and changing it needs root, and the shared RTX 3090 is
> in use by other workloads. Written against the
> [production-stack values.yaml](https://github.com/vllm-project/production-stack/blob/main/helm/values.yaml)
> (chart `vllm-stack` 0.1.14) and the
> [KServe LLMInferenceService docs](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview)
> (v0.20), accessed 2026-11-07. Checked by rendering with `helm template` and by
> `kubectl apply --dry-run=server` on the kind cluster (KServe: against its v0.20.0 CRD).

```yaml
# ch20/01_main-chapter-code/gpu/production-stack-values.yaml (excerpt)
servingEngineSpec:
  runtimeClassName: nvidia
  modelSpec:
    - name: qwen3-8b
      repository: vllm/vllm-openai
      tag: v0.33.0                    # pin the engine like any other dependency
      modelURL: Qwen/Qwen3-8B
      requestGPU: 1
      pvcStorage: 40Gi                # weights are downloaded once and survive restarts
      vllmConfig:
        enablePrefixCaching: true     # the FAQ system prompt is a shared prefix
        maxModelLen: 16384
        gpuMemoryUtilization: 0.90
  startupProbe:                       # 10 minutes to load before liveness may kill it
    periodSeconds: 10
    failureThreshold: 60
routerSpec:
  routingLogic: prefixaware           # same prompt prefix -> same engine -> KV cache hits
```

```yaml
# ch20/01_main-chapter-code/gpu/kserve-llminferenceservice.yaml (excerpt)
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
spec:
  model: {uri: hf://Qwen/Qwen3-8B, name: Qwen/Qwen3-8B}
  replicas: 1
  template:
    containers:
      - name: main
        image: vllm/vllm-openai:v0.33.0
        resources: {limits: {nvidia.com/gpu: "1", cpu: "6", memory: 24Gi}}
  router: {gateway: {}, route: {}, scheduler: {}}
```

## Results

Time to go back to the Requirements list. The table puts each target next to what we measured,
with v0's numbers where we have them:

| Requirement | Target | Measured |
|---|---|---|
| Images on Compose and Kubernetes | both | ✅ Compose `prod` healthy in 5.8 s; 3-node kind cluster ready in 62 s |
| Follow-up turns see the full history | 100% | v0: 11/30 (turn 2), 3/30 (turn 3) → hardened: 30/30, 30/30 |
| Canary split, sticky, reversible | ±5 points | 11/120 (9.2%) at 10%, 58/120 (48.3%) at 50%; 11/11 kept v2 through the ramp; rollback one command (9 s) |
| Eval gate blocks a known regression | blocks | PR 2 blocked (86.0%, flipped `s1`); PR 1 passed with a 47% shorter planner prompt |
| Failed requests during a rolling deploy | 0 | v0: 136/879 (15.5%) → hardened: 0/760 |
| Image | < 250 MB, non-root, starts offline | 1,206 MB, root, crashes offline → 191 MB, uid 10001, serving offline in 2.0 s |
| Scaling | 2 → up to 6 and back | 2 → 6 within 61 s of load; back to 2 at 329 s; 0/2,534 failed |
| Node drains | ≥ 1 chat pod serving | drain blocked on the last pod by the PDB; 0/252 requests failed |
| Hosted spend | < $1 | $0.053 for the final notebook run (sum of `usage.cost`) |

## Exercises

These exercises push on the parts of the chapter where our choices were simplest: the shutdown
sleep, the CPU signal, the gate and the canary verdict.

1. Remove the `preStop` sleep from `k8s/base/chat.yaml` (keep everything else) and rerun the
   rolling-restart cell. How many requests fail now, and which `status` do they have? Then set
   `--timeout-graceful-shutdown 1` in the Dockerfile instead and compare.
2. Replace the CPU target in `chat-scaling.yaml` with a custom metric: count in-flight streams in
   the chat service, expose it, and scale on it with KEDA or the Prometheus adapter (Chapter 11
   has the Prometheus side). Does the HPA react faster than the 30 s CPU lag we measured?
3. Write a prompt v3 that breaks one of the six gate checks (for example, let the bot "apply"
   goodwill discounts). Run `ch20_eval_gate.py --chat-url` against it locally, then try it as a
   10% canary. Which of the two catches it, and how many customers saw it first?
4. Make the canary decision statistical: in `ch20_canary.verdict`, replace the fixed
   thresholds with a two-proportion test on error rates and a bootstrap interval on p95 latency.
   How many canary requests do you need before it can tell a 2-point error-rate increase from noise?

## What we learned

- A deploy is a sequence of measurable events, not a YAML file. Measure memory, rolling
  restarts, drains and scaling under load before you trust a manifest.
- Multi-stage builds with dependencies first keep images small and rebuilds instant; baking
  every runtime download (here, the tokenizer) into the image is what lets a pod start without
  internet access.
- Anything a replica keeps in memory is lost to the next request on another replica. Session
  state goes to a shared store before the second replica exists.
- Liveness checks the process, readiness checks dependencies. Mixing them turns a database
  outage into a restart storm.
- Graceful shutdown is a race between endpoint removal and `SIGTERM`; a `preStop` sleep plus a
  grace period long enough for the longest stream wins it.
- CPU-based autoscaling works for a streaming proxy only because CPU happens to track
  concurrency; it lags by tens of seconds, and for GPU model servers it's the wrong signal.
- An eval gate with a fixed bar is only as sharp as its sample. Repeats and an item-level
  comparison with main catch regressions that an average hides.
- Treat a prompt like a release: version it, canary it on a sticky hash of the session,
  analyse the canary from per-request logs, and keep rollback one command away.

We now ship the bots like any other service, and a deploy no longer costs us a single request.
But every load test here used at most 40 concurrent streams against a fake model. Chapter 21 asks
what happens at a million monthly users: it load-tests the Chapter 1 API, finds where it breaks,
and adds rate limits, failover and job queues to fix each bottleneck.

## References

- "API, ChatGPT & Sora Facing Issues" (incident write-up, December 11, 2024) — OpenAI Status. https://status.openai.com/incidents/01JMYB483C404VMPCW726E8MET
- "Expanding on what we missed with sycophancy" — OpenAI, 2025-05-02. https://openai.com/index/expanding-on-sycophancy/
- "A postmortem of three recent issues" — Anthropic Engineering, 2025-09-17. https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues
- "Evolving GitHub Copilot's next edit suggestions through custom model training" — GitHub Blog. https://github.blog/ai-and-ml/github-copilot/evolving-github-copilots-next-edit-suggestions-through-custom-model-training/
- kind quick start (v0.33.0) — Kubernetes SIGs, accessed 2026-11-07. https://kind.sigs.k8s.io/docs/user/quick-start/
- Install and set up kubectl on Linux — Kubernetes docs, accessed 2026-11-07. https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/
- Installing Helm — Helm docs (v4.3.1), accessed 2026-11-07. https://helm.sh/docs/intro/install/
- metrics-server v0.9.1 — Kubernetes SIGs. https://github.com/kubernetes-sigs/metrics-server
- Horizontal Pod Autoscaling — Kubernetes docs, accessed 2026-11-07. https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- Disruptions (PodDisruptionBudget) — Kubernetes docs. https://kubernetes.io/docs/concepts/workloads/pods/disruptions/
- Configure liveness, readiness and startup probes — Kubernetes docs. https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
- Deployments (rollout, undo, revision history) — Kubernetes docs. https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- Secrets — Kubernetes docs. https://kubernetes.io/docs/concepts/configuration/secret/
- Multi-stage builds — Docker docs. https://docs.docker.com/build/building/multi-stage/
- Using uv in Docker — Astral docs, accessed 2026-11-07. https://docs.astral.sh/uv/guides/integration/docker/
- Using profiles with Compose — Docker docs. https://docs.docker.com/compose/how-tos/profiles/
- Use Docker service containers — GitHub Docs. https://docs.github.com/en/actions/tutorials/use-containerized-services/use-docker-service-containers
- Managing environments for deployment — GitHub Docs. https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments
- Control the concurrency of workflows and jobs — GitHub Docs. https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency
- actionlint 1.7.12 — rhysd. https://github.com/rhysd/actionlint
- vLLM production-stack (Helm chart `vllm-stack` 0.1.14) — vLLM project. https://github.com/vllm-project/production-stack
- Understanding LLMInferenceService — KServe docs (v0.20), accessed 2026-11-07. https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview
- NVIDIA/nvkind — NVIDIA. https://github.com/NVIDIA/nvkind
- Best practices for autoscaling LLM inference workloads with GPUs — Google Cloud (GKE docs). https://cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling
- kserve/kserve #5062, liveness probe kills large models before they finish loading. https://github.com/kserve/kserve/issues/5062
- vllm-project/vllm #58544, Helm chart liveness probe restarts pods during model load. https://github.com/vllm-project/vllm/issues/58544
- "KiND – How I Wasted a Day Loading Local Docker Images" — iximiuz. https://iximiuz.com/en/posts/kubernetes-kind-load-docker-image/
- kubernetes/ingress-nginx #14090, SSE streams closed after an nginx reload. https://github.com/kubernetes/ingress-nginx/issues/14090
- "Ingress NGINX retirement" — Kubernetes Blog, 2025-11-11. https://kubernetes.io/blog/2025/11/11/ingress-nginx-retirement/
- "Graceful Kubernetes deployments" — Rakuten Engineering. https://engineering.rakuten.today/post/graceful-k8s-delpoyments/
- kubernetes/kubernetes #105703, existing connections survive endpoint removal. https://github.com/kubernetes/kubernetes/issues/105703
