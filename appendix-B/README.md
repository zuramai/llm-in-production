# Appendix B: Glossary

The chapters define each new term where it first appears. But the book is long, and a word from
Chapter 4 may come back in Chapter 17 when you no longer remember it. This glossary is the place
to look it up. It holds every term the book uses that a backend engineer might not know yet.
Each entry gives a plain-language definition, in the sense the chapters use it. It also links to
the sections where the term is introduced and put to work. The first link is usually the best
place to start.

How to use it:

- Terms are alphabetical, grouped by letter. Symbols and backticks are ignored when sorting, so
  `finish_reason` sits under F and `p50, p95` under P.
- Closely related terms share one entry (*TTFT, TPOT, total latency*). If you can't find a term,
  search this page for it: it's probably in the definition of its neighbor.
- Words in *italics* inside a definition have their own entry.
- Some words mean different things in different chapters (*token*, *thread*, *fan-out*,
  *recall*, *metric*). Each entry says which sense it covers and points to the other.
- Numbers quoted here come from the chapters and keep the chapters' dates: prices and versions
  are as of 2026-11.

[A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [J](#j) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [V](#v) · [W](#w)

## A

**Agent**  
An LLM in a loop with tools. Each turn, the model reads the conversation so far. Then it either
calls one or more tools or stops. Our code runs the tools and appends the results. What makes it an
agent is that the *model* picks the next step, not our code. Compare *workflow*.  
*See:* [Ch 5 › From pipeline to agent: who decides the next step?](../ch05/README.md#from-pipeline-to-agent-who-decides-the-next-step) · [Ch 10 › Workflow or agent?](../ch10/README.md#workflow-or-agent) · [Ch 11 › An agent is a loop](../ch11/README.md#an-agent-is-a-loop) · [Ch 5 › 1. Give the model tools: a plain tool loop](../ch05/README.md#1-give-the-model-tools-a-plain-tool-loop)

**Alias (model alias)**  
A stable name such as `fast` that apps send to a *gateway* instead of a real model ID. The gateway decides which real model the alias means today, so swapping models is a config change, not a code change.  
*See:* [Ch 15 › A gateway is a reverse proxy that speaks the OpenAI API](../ch15/README.md#a-gateway-is-a-reverse-proxy-that-speaks-the-openai-api)

**Allowlist (tool allowlist)**  
A list of what is allowed; everything else is refused. In Chapter 13 each role (support agent,
customer) gets a fixed set of tools. The tool layer enforces it twice: for what the model *sees* and
for what it can *execute*. Chapter 11 uses the same idea to decide which log fields may leave the
process.  
*See:* [Ch 13 › 4. A tool allowlist per role, and a call budget](../ch13/README.md#4-a-tool-allowlist-per-role-and-a-call-budget) · [Ch 11 › The telemetry plumbing](../ch11/README.md#the-telemetry-plumbing)

**Approval gate**  
A point where an action the agent proposes (a refund) waits for a human with the right role to
approve it. The model may *propose*. Ordinary code *validates*. A human *decides*. Compare
*human-in-the-loop*.  
*See:* [Ch 9 › Who holds the privilege](../ch09/README.md#who-holds-the-privilege) · [Ch 9 › 4. Side effects: propose, don't perform](../ch09/README.md#4-side-effects-propose-dont-perform) · [Ch 10 › 1. Pause before the side effect](../ch10/README.md#1-pause-before-the-side-effect)

**AST guard**  
A safety check for model-written SQL. It parses the SQL into an abstract syntax tree (with sqlglot)
and rejects anything that isn't a single read-only query over allowed tables. A parser knows what a
statement *is*; a regex blocklist doesn't. It's still only one layer.  
*See:* [Ch 6 › 3. An AST guard](../ch06/README.md#3-an-ast-guard) · [Ch 13 › 3. Privilege separation](../ch13/README.md#3-privilege-separation)

## B

**Backoff (exponential)**  
Waiting longer before each retry: 0.5 s, 1 s, 2 s, 4 s... so a struggling provider gets room to recover. Usually paired with *jitter*. The OpenAI SDK does this by default for 429s and 5xx.  
*See:* [Ch 21 › Retries, backoff and jitter](../ch21/README.md#retries-backoff-and-jitter) · [Ch 1 › 4. Survive the network: timeouts, retries, fallbacks](../ch01/README.md#4-survive-the-network-timeouts-retries-fallbacks)

**Business time zone**  
The time zone the business runs on (New York for ShopFlow), as opposed to the UTC the database stores. "Today", "yesterday" and every *time grain* must be cut in this zone, or "yesterday" starts at 7 PM.  
*See:* [Ch 7 › Time grains and the business time zone](../ch07/README.md#time-grains-and-the-business-time-zone) · [Ch 6 › 1. A semantic context instead of raw DDL](../ch06/README.md#1-a-semantic-context-instead-of-raw-ddl) · [Ch 22 › G1 Timezone "today"](../ch22/README.md#g1-timezone-today) · [Ch 22 › G4 Aware datetime arithmetic ignores DST](../ch22/README.md#g4-aware-datetime-arithmetic-ignores-dst)

## C

**Calibrated confidence**  
A confidence score is calibrated when it matches reality over many cases: the answers marked 0.9 are right about 90% of the time. A model's self-reported "confidence 0.99" is a number it wrote, not a measured probability, and in our runs it often barely separated right from wrong.  
*See:* [Ch 11 › Hypotheses, evidence and confidence](../ch11/README.md#hypotheses-evidence-and-confidence) · [Ch 12 › 3. Snap, dedupe, filter](../ch12/README.md#3-snap-dedupe-filter) · [Ch 22 › E2 Self-reported confidence is uncalibrated](../ch22/README.md#e2-self-reported-confidence-is-uncalibrated)

**Calibration (of a judge)**  
Checking an *LLM judge* against human labels on the same answers. Then we adjust its rubric and
reference until they agree. We require quadratic-weighted *Cohen's kappa* ≥ 0.8. An uncalibrated
judge is just another opinion.  
*See:* [Ch 3 › 3. A rubric and a reference: calibrating the judge](../ch03/README.md#3-a-rubric-and-a-reference-calibrating-the-judge) · [Ch 3 › Agreement, and why accuracy isn't enough: Cohen's kappa](../ch03/README.md#agreement-and-why-accuracy-isnt-enough-cohens-kappa)

**Canary**  
Sending a new version (of an image or a prompt) to a small share of real traffic first. We compare
it with the stable version on live metrics, then ramp up or roll back. An *eval gate* catches what
you thought to test. A canary catches what you didn't, and limits how many users it can hurt.  
*See:* [Ch 20 › Shipping a prompt: canary, and the gate before it](../ch20/README.md#shipping-a-prompt-canary-and-the-gate-before-it) · [Ch 20 › 6. A prompt canary](../ch20/README.md#6-a-prompt-canary)

**Cascade**  
Asking a cheap model first and escalating to an expensive one only when needed. The hard part is deciding "when needed" cheaply; we use a *verifier*.  
*See:* [Ch 15 › A cascade spends the strong model only where it matters](../ch15/README.md#a-cascade-spends-the-strong-model-only-where-it-matters) · [Ch 15 › 7. A cheap → strong cascade with a verifier](../ch15/README.md#7-a-cheap--strong-cascade-with-a-verifier)

**Chat Completions**  
The OpenAI HTTP API shape (`client.chat.completions.create`). It takes a list of messages and
returns the next one. It's the common language of the book: OpenRouter, Ollama, vLLM, llama.cpp and
LiteLLM all implement it. So one code path reaches any provider. Compare *Responses API*.  
*See:* [Ch 1 › The model landscape (as of 2026-11-08)](../ch01/README.md#the-model-landscape-as-of-2026-11-09) · [Ch 15 › A gateway is a reverse proxy that speaks the OpenAI API](../ch15/README.md#a-gateway-is-a-reverse-proxy-that-speaks-the-openai-api)

**Chat template, completion-only loss**  
The model-specific recipe that turns a list of chat messages into the single string the model
actually reads. It uses special tokens such as `<|im_start|>`, and engines apply it on every
request. For fine-tuning, three rules follow. Train on exactly the string you'll serve. Compute the
loss only on the answer (*completion-only loss*), so the model doesn't learn to write tickets. And
include the end-of-turn token, or the model never learns to stop.  
*See:* [Ch 18 › What the model actually trains on](../ch18/README.md#what-the-model-actually-trains-on)

**Checkpoint, checkpointer**  
A checkpoint is a snapshot of an agent's state, saved after every step, like a game's autosave. The
checkpointer is the part that stores them (in Postgres for us). With one, a run is a row that any
process can resume, not a Python call stack.  
*See:* [Ch 10 › Durable execution: checkpoints and threads](../ch10/README.md#durable-execution-checkpoints-and-threads) · [Ch 10 › 2. Survive restarts: the Postgres checkpointer](../ch10/README.md#2-survive-restarts-the-postgres-checkpointer) · [Ch 5 › LangGraph in five words: state, nodes, edges, checkpoints](../ch05/README.md#langgraph-in-five-words-state-nodes-edges-checkpoints) · [Ch 5 › 5. Survive a failure halfway through: checkpoints](../ch05/README.md#5-survive-a-failure-halfway-through-checkpoints)

**Chunk**  
A piece of a document small enough that several fit in a prompt. RAG retrieves chunks, not whole
documents. *Fixed windows* cut every N tokens, with some overlap. *Heading-aware* chunks follow the
document's sections and carry their heading path.  
*See:* [Ch 4 › Chunks](../ch04/README.md#chunks) · [Ch 4 › 3. Heading-aware chunks](../ch04/README.md#3-heading-aware-chunks)

**Circuit breaker**  
A switch in front of a dependency. After repeated failures (3 in a row, in Chapter 21) it stops
sending traffic there. After a pause it lets one trial request through. It turns a slow, timing-out
dependency into a fast, clear failure. That is what makes *failover* quick.  
*See:* [Ch 21 › 5. Fail over to another provider](../ch21/README.md#5-fail-over-to-another-provider)

**Closed-loop benchmark**  
A load test that keeps exactly `c` requests in flight: a new one starts only when one finishes. It
is simple and repeatable. But it can't show what happens when requests arrive faster than the server
can handle. An *open-loop* benchmark sends at a fixed arrival rate instead. *Goodput* counts only
the requests that met their latency target.  
*See:* [Ch 16 › What we measure](../ch16/README.md#what-we-measure)

**Cohen's kappa (κ)**  
How much two raters (a judge and a human) agree *beyond what chance would give*. 1 is perfect
agreement; 0 is chance level. Plain percent agreement can look good if a rater always gives the most
common score; kappa can't be fooled that way. The *quadratic-weighted* variant punishes being off by
two more than being off by one.  
*See:* [Ch 3 › Agreement, and why accuracy isn't enough: Cohen's kappa](../ch03/README.md#agreement-and-why-accuracy-isnt-enough-cohens-kappa)

**Compaction (of tool outputs)**  
Shrinking long tool results before they go back to the model. Here we prune the JSON by structure:
the first N items of each list, then "N more omitted". The full output stays in the checkpoint; only
the *view* sent to the model gets shorter. Compact once and never rewrite, or you lose the *prompt
cache*.  
*See:* [Ch 10 › 5. Compact long tool outputs](../ch10/README.md#5-compact-long-tool-outputs)

**Concurrent streams**  
How many streamed answers are open at the same moment. Each answer stays open for seconds. So for an
LLM app, this number, not requests per second, sizes servers and pools. *Little's law* computes
it.  
*See:* [Ch 21 › Little's law: the one formula of capacity planning](../ch21/README.md#littles-law-the-one-formula-of-capacity-planning) · [Ch 21 › Napkin math for a million monthly users](../ch21/README.md#napkin-math-for-a-million-monthly-users)

**ConfigMap, Secret (Kubernetes)**  
The Kubernetes objects that inject plain settings (ConfigMap) and credentials (Secret) into pods as environment variables, so one image runs everywhere. A Secret is only base64-encoded, not encrypted, by default.  
*See:* [Ch 20 › Configuration lives outside the image](../ch20/README.md#configuration-lives-outside-the-image)

**Connection pool**  
A fixed set of reusable connections (to a database, Redis or the LLM provider). Requests borrow one
and give it back. A full pool is a silent queue: no errors, just waiting. So every pool needs a size
worked out with *Little's law*, and a metric.  
*See:* [Ch 21 › Three queues inside one Python process](../ch21/README.md#three-queues-inside-one-python-process) · [Ch 21 › 2. Size the connection pool on purpose](../ch21/README.md#2-size-the-connection-pool-on-purpose)

**Constrained decoding**  
At each step, the model may only pick tokens that keep the output valid against a grammar. The
grammar is compiled from our JSON Schema. It guarantees the *shape*, not the truth. And it can't
finish a reply that runs out of `max_tokens`.  
*See:* [Ch 2 › Constrained decoding, in one picture](../ch02/README.md#constrained-decoding-in-one-picture)

**Container image**  
A file-system snapshot plus a start command: Python, dependencies, our code. It's the unit we test, ship and roll back, like a compiled binary. A *multi-stage build* builds in one stage and copies only the result into a clean final stage.  
*See:* [Ch 20 › An image is a frozen environment; a multi-stage build keeps only what runs](../ch20/README.md#an-image-is-a-frozen-environment-a-multi-stage-build-keeps-only-what-runs) · [Ch 20 › 1. A multi-stage image that starts offline](../ch20/README.md#1-a-multi-stage-image-that-starts-offline)

**Context window**  
The model's working memory: the most tokens (prompt plus answer) one request can hold. You pay for
every token in it on every turn, and each one delays the first token. So we set our own, much
smaller *token budget*.  
*See:* [Ch 1 › Context window, and why we set a smaller budget](../ch01/README.md#context-window-and-why-we-set-a-smaller-budget) · [Ch 10 › 5. Compact long tool outputs](../ch10/README.md#5-compact-long-tool-outputs) · [Ch 22 › A6 Context overflow is silent](../ch22/README.md#a6-context-overflow-is-silent)

**Continuous batching**  
Rebuilding the batch at every decode step. Finished requests leave, and waiting ones join after
their *prefill*. So the GPU never sits idle waiting for the longest answer (static batching waits
for the whole batch). One read of the weights serves every request in the batch, and this is where
self-hosted throughput comes from. vLLM, SGLang and llama-server all do it. Ollama does it only up
to `OLLAMA_NUM_PARALLEL` requests per model.  
*See:* [Ch 16 › Continuous batching](../ch16/README.md#continuous-batching) · [Ch 17 › Continuous batching](../ch17/README.md#continuous-batching) · [Ch 17 › 2. Batching: more tokens per weight read](../ch17/README.md#2-batching-more-tokens-per-weight-read)

**Corrective RAG (CRAG)**  
Grading what the retriever returned *before* answering, and rewriting the query and searching again when it isn't good enough. A retriever always returns *something*; this catches the times it's irrelevant.  
*See:* [Ch 5 › Grade, then rewrite: corrective RAG](../ch05/README.md#grade-then-rewrite-corrective-rag) · [Ch 5 › 3. Stop the grader from chasing evidence that doesn't exist](../ch05/README.md#3-stop-the-grader-from-chasing-evidence-that-doesnt-exist)

**Cosine similarity, cosine distance**  
How close two *embeddings* point: 1.0 means the same direction, 0 means unrelated (distance is one minus similarity). Close in meaning isn't the same as having the same answer, which is what the *semantic cache* stumbles over.  
*See:* [Ch 4 › Embeddings and the HNSW index](../ch04/README.md#embeddings-and-the-hnsw-index) · [Ch 15 › Four caches, four different things](../ch15/README.md#four-caches-four-different-things)

**Cross-encoder (reranker)**  
A model that reads the question and one passage *together* and gives a relevance score. It is far
more accurate than comparing two separate embeddings. It is also far too slow to run on every chunk.
So it reranks only the top candidates from a fast first search.  
*See:* [Ch 4 › Cross-encoder reranking](../ch04/README.md#cross-encoder-reranking) · [Ch 4 › 2. A cross-encoder reranker, on the GPU](../ch04/README.md#2-a-cross-encoder-reranker-on-the-gpu)

**CTE (common table expression)**  
A named subquery in a SQL `WITH` clause. The AST guard has to look inside CTEs, because Postgres
allows `DELETE ... RETURNING` there. The semantic layer adds up each entity in its own CTE to avoid
*fan-out*.  
*See:* [Ch 6 › 3. An AST guard](../ch06/README.md#3-an-ast-guard) · [Ch 7 › Fan-out: why a semantic layer refuses some questions](../ch07/README.md#fan-out-why-a-semantic-layer-refuses-some-questions)

## D

**Data leakage (train/test overlap), dedupe**  
When examples from the test (or validation) set also appear in the training set. A tuned model can
then score on them from memory, and the test number means nothing. Chapter 18's first split was made
by ticket id, not by text. It put 43 test tickets word for word into the training file. Chapter 19's
validators caught it. Chapter 18 then fixed the split and reports the unavoidable identical-text
tickets as a separate segment. Deduping (dropping exact repeats) keeps the leak from coming back. It
also cuts the bill, because you pay for every duplicate on every *epoch*.  
*See:* [Ch 18 › The leak we shipped, and the fix](../ch18/README.md#the-leak-we-shipped-and-the-fix) · [Ch 19 › Measuring v0](../ch19/README.md#measuring-v0) · [Ch 19 › 1. Fix the split at the source, keep a dedupe guard](../ch19/README.md#1-fix-the-split-at-the-source-keep-a-dedupe-guard)

**Dataset experiment (Langfuse)**  
A run of one prompt or app version over every item of a stored dataset (our eval set). Each item
gets a trace and scores, so two versions can be compared side by side. Langfuse used to call it a
*dataset run*. It's the *eval*, living inside the observability tool.  
*See:* [Ch 14 › 7. The eval as a dataset experiment](../ch14/README.md#7-the-eval-as-a-dataset-experiment)

**Defense in depth**  
Stacking independent safety layers (structured plan, AST guard, read-only role, ...) so that a bug in one doesn't become an incident. No single layer is trusted to be enough.  
*See:* [Ch 6 › Defense in depth](../ch06/README.md#defense-in-depth) · [Ch 13 › Guardrails are layers, and each one has a job](../ch13/README.md#guardrails-are-layers-and-each-one-has-a-job)

**Delimiting (untrusted text)**  
Wrapping text from strangers in tags (`<ticket>...</ticket>`) and escaping `<` and `>` inside it.
Then we tell the model never to follow instructions found there. It helps against *prompt injection*
but doesn't solve it. In Chapter 2 it fixed one model and not the other.  
*See:* [Ch 2 › 5. Prompt injection inside the ticket](../ch02/README.md#5-prompt-injection-inside-the-ticket) · [Ch 9 › 5. Prompt injection: tickets are data](../ch09/README.md#5-prompt-injection-tickets-are-data)

**Deployment, Service, pod (Kubernetes)**  
A *pod* is one running instance of our containers. A *Deployment* keeps N identical pods running and replaces them gradually on change; a *Service* gives them one stable name and load-balances over the ready ones.  
*See:* [Ch 20 › Kubernetes in seven objects](../ch20/README.md#kubernetes-in-seven-objects) · [Ch 20 › What happens when a pod is replaced](../ch20/README.md#what-happens-when-a-pod-is-replaced)

**Dimension (semantic layer)**  
A column you can group or filter a metric by, such as `customer.segment`. In a semantic layer a dimension is reachable only along many-to-one joins, which is how it prevents *fan-out*.  
*See:* [Ch 7 › What a semantic layer is](../ch07/README.md#what-a-semantic-layer-is)

**Disaggregated prefill and decode**  
Running *prefill* and *decode* on separate GPU pools and moving the *KV cache* between them. Prefill
is compute-bound and bursty; decode is limited by memory bandwidth and steady. Splitting them means
a big prompt doesn't stall everyone else's tokens. It is a multi-GPU technique; the book describes
it but didn't run it.  
*See:* [Ch 17 › Disaggregated prefill and decode](../ch17/README.md#disaggregated-prefill-and-decode)

**Docker Compose**  
A YAML file that describes several containers (app, database, gateway, tracing backend). `docker
compose up` starts them together on one host. It is the book's default way to run a chapter's stack.
Kubernetes is the multi-host version of the same idea.  
*See:* [Ch 20 › On one host: Compose](../ch20/README.md#on-one-host-compose) · [Ch 22 › F5 Compose env precedence and `${VAR:?}` on `down`](../ch22/README.md#f5-compose-env-precedence-and-var-on-down)

**Durable execution**  
Running an agent so that its progress survives crashes and restarts: state is saved after every step (*checkpoint*), and a run can be resumed by any process, hours or days later.  
*See:* [Ch 10 › Durable execution: checkpoints and threads](../ch10/README.md#durable-execution-checkpoints-and-threads)

## E

**Embedding**  
A list of numbers (768 for `nomic-embed-text`) that places a text in a "meaning space". Texts about
the same thing land close together, whatever words they use. Embeddings power vector search and the
*semantic cache*.  
*See:* [Ch 4 › Embeddings and the HNSW index](../ch04/README.md#embeddings-and-the-hnsw-index) · [Ch 15 › Four caches, four different things](../ch15/README.md#four-caches-four-different-things)

**Entity (semantic layer)**  
A table and its key, plus its many-to-one joins and its time column, such as `order` dated by `placed_at`. Measures are defined on one entity.  
*See:* [Ch 7 › What a semantic layer is](../ch07/README.md#what-a-semantic-layer-is)

**Epoch**  
One pass over the whole training set during *fine-tuning*. Managed providers bill training as tokens in the file × number of epochs, so three epochs cost three times one.  
*See:* [Ch 19 › What "managed" fine-tuning hands over](../ch19/README.md#what-managed-fine-tuning-hands-over)

**Escalation**  
Handing a case to something more capable when the first attempt fails. That can be a stronger model
reading the page fresh (Chapter 8), the expensive model in a *cascade* (Chapter 15), or a human
queue. In Chapter 8, a stronger model that was shown the weak model's answer kept its mistake.  
*See:* [Ch 8 › 4. Validate, repair, escalate](../ch08/README.md#4-validate-repair-escalate) · [Ch 15 › A cascade spends the strong model only where it matters](../ch15/README.md#a-cascade-spends-the-strong-model-only-where-it-matters)

**Eval**  
A run of your system on a fixed set of inputs, with the outputs scored. It's a unit test whose assertion is fuzzy: we check *properties* of the answer ("mentions 30 days") rather than an exact string.  
*See:* [Ch 3 › An eval is a unit test with a score instead of a boolean](../ch03/README.md#an-eval-is-a-unit-test-with-a-score-instead-of-a-boolean)

**Eval gate, regression gate**  
A CI check that runs the eval set and fails the build below a threshold. A gate is only reassuring if it *can* fail: test it with deliberately broken prompts and set the threshold from what it catches.  
*See:* [Ch 3 › 2. A regression gate that can actually fail](../ch03/README.md#2-a-regression-gate-that-can-actually-fail) · [Ch 20 › Shipping a prompt: canary, and the gate before it](../ch20/README.md#shipping-a-prompt-canary-and-the-gate-before-it) · [Ch 20 › 5. An eval gate in front of every image](../ch20/README.md#5-an-eval-gate-in-front-of-every-image) · [Ch 22 › E3 Small eval gates are noisy](../ch22/README.md#e3-small-eval-gates-are-noisy)

**Event loop**  
The single thread in an async Python server that juggles thousands of waiting requests. This only
works if every request `await`s while it waits. One blocking call inside `async def` (a sync HTTP
client, `time.sleep`) stalls everything, health checks included.  
*See:* [Ch 21 › Three queues inside one Python process](../ch21/README.md#three-queues-inside-one-python-process) · [Ch 21 › 1. Off the event loop and onto more cores](../ch21/README.md#1-off-the-event-loop-and-onto-more-cores)

**Exact cache**  
A cache keyed on the exact request (model, messages, parameters): a byte-identical request gets the stored answer back without calling the model. Cheap and safe, but any change in wording misses. Compare *semantic cache* and *prompt caching*.  
*See:* [Ch 15 › Four caches, four different things](../ch15/README.md#four-caches-four-different-things) · [Ch 15 › 4. Exact cache in Redis](../ch15/README.md#4-exact-cache-in-redis)

**Excessive agency**  
OWASP's name for an LLM application that can do more than its task needs: more tools, more permissions, more autonomy. It ranks #3 in the 2026 OWASP Top 10 for LLM Applications.  
*See:* [Ch 13 › Mapping to the OWASP Top 10 for LLM Applications 2026](../ch13/README.md#mapping-to-the-owasp-top-10-for-llm-applications-2026)

**Execution accuracy**  
Scoring text-to-SQL by *running* the generated query and the gold query and comparing result rows (ignoring order and column names), instead of comparing SQL text. There are many correct ways to write a query.  
*See:* [Ch 6 › Execution accuracy, not exact match](../ch06/README.md#execution-accuracy-not-exact-match)

**Exfiltration**  
Getting private data out to an attacker. In LLM apps the channel is often innocent-looking output: a rendered markdown image whose URL carries the data is fetched automatically, with zero clicks.  
*See:* [Ch 13 › The lethal trifecta](../ch13/README.md#the-lethal-trifecta) · [Ch 13 › 1. A markdown and URL sanitizer](../ch13/README.md#1-a-markdown-and-url-sanitizer)

## F

**Failover**  
Switching to another provider when the main one fails. In Chapter 21 we fail over only *before the
first token*, because after that the user has already seen half an answer. We also fail over only on
errors worth retrying, with a *circuit breaker* per provider.  
*See:* [Ch 21 › 5. Fail over to another provider](../ch21/README.md#5-fail-over-to-another-provider)

**Faithfulness**  
Whether a generated answer sticks to the retrieved text, as opposed to *correctness* against a human reference. An answer can be faithful to a wrong passage, or correct from the model's memory and unfaithful to the context.  
*See:* [Ch 4 › Retrieval-augmented generation](../ch04/README.md#retrieval-augmented-generation)

**Fallback**  
A degraded alternative used when the primary call fails, such as answering from a local model on timeout or rate limit. A fallback is only as good as its latency: one that takes a minute is an outage with extra steps.  
*See:* [Ch 1 › 4. Survive the network: timeouts, retries, fallbacks](../ch01/README.md#4-survive-the-network-timeouts-retries-fallbacks) · [Ch 15 › 3. Fallbacks, tested with a game day](../ch15/README.md#3-fallbacks-tested-with-a-game-day)

**False positive**  
A finding that isn't real: for the review bot, a posted comment that matches no actual bug. Every false positive teaches developers to ignore the bot.  
*See:* [Ch 12 › Recall, false positives, and why a planted-bug eval](../ch12/README.md#recall-false-positives-and-why-a-planted-bug-eval)

**Fan-out (join)**  
A silent analytics bug. You join a "one" table to a "many" table, then sum a column from the "one"
side. Each value gets counted once per child row. On ShopFlow it doubled gross revenue. (The book
also says "fan out" for running many calls in parallel; context tells them apart.)  
*See:* [Ch 7 › Fan-out: why a semantic layer refuses some questions](../ch07/README.md#fan-out-why-a-semantic-layer-refuses-some-questions)

**Fencing**  
Making a slow worker's late write fail if another worker has since re-claimed its job. The write
checks that the worker still owns the *lease* (same `locked_by` and attempt number).  
*See:* [Ch 21 › 6. Async work: a Postgres queue with `FOR UPDATE SKIP LOCKED`](../ch21/README.md#6-async-work-a-postgres-queue-with-for-update-skip-locked)

**Few-shot prompt**  
A prompt that shows worked examples (an input and the expected output) before the real input. In
Chapter 19 a 22-example prompt beat the plain rulebook by about 20 points on priority. It is the
baseline a *fine-tuned* model has to beat. Examples teach exactly what they cover, and little beyond
it.  
*See:* [Ch 19 › The "before" number](../ch19/README.md#the-before-number) · [Ch 18 › Measuring v0](../ch18/README.md#measuring-v0)

**Field-level accuracy**  
Scoring an extraction one field at a time (supplier, invoice number, totals, line items) against ground truth, instead of a single "accuracy". The strictest level, every field right, means zero edits needed.  
*See:* [Ch 8 › Field-level accuracy](../ch08/README.md#field-level-accuracy)

**Fine-tuning (managed fine-tuning)**  
Training an existing model further on your own input → output examples. It learns a task's format
and policy without a long prompt. It changes the weights, not the prompt. It's a poor way to add
facts that change; that's *RAG*'s job. Chapter 18 does it on our own GPU (*LoRA*, *QLoRA*). In
*managed* fine-tuning (Chapter 19) you upload a training file and the provider trains on its
hardware. You get back a model id you can call but not download. It lives only as long as its base
model (see *lease*). See also *SFT*, *epoch*.  
*See:* [Ch 18 › Fine-tuning changes the weights, not the prompt](../ch18/README.md#fine-tuning-changes-the-weights-not-the-prompt) · [Ch 18 › Prompt, RAG or fine-tune?](../ch18/README.md#prompt-rag-or-fine-tune) · [Ch 19 › What "managed" fine-tuning hands over](../ch19/README.md#what-managed-fine-tuning-hands-over) · [Ch 19 › A job's life](../ch19/README.md#a-jobs-life) · [Ch 19 › A tuned model is a lease](../ch19/README.md#a-tuned-model-is-a-lease)

**`finish_reason`**  
The field in a response that says why generation stopped: `stop` (finished), `length` (hit
`max_tokens`), `tool_calls` (wants a tool) or `content_filter` (blocked). Check it before parsing. A
`length` reply is half an answer, and retrying it fails the same way.  
*See:* [Ch 2 › 3. The truncation trap](../ch02/README.md#3-the-truncation-trap) · [Ch 2 › 2. Retry only what a retry can fix](../ch02/README.md#2-retry-only-what-a-retry-can-fix) · [Ch 9 › A tool call is a request, not an action](../ch09/README.md#a-tool-call-is-a-request-not-an-action) · [Ch 22 › A1 `finish_reason == "length"` truncates JSON](../ch22/README.md#a1-finish_reason--length-truncates-json) · [Ch 22 › B3 Safety filter: empty reply, `finish_reason="content_filter"`](../ch22/README.md#b3-safety-filter-empty-reply-finish_reasoncontent_filter)

**Full-text search**  
Word search built into Postgres. `to_tsvector` reduces text to word stems, a `tsquery` matches them,
and `ts_rank_cd` scores the matches. It is blind to meaning. But it is exact about strings like
`payments.primary_provider`, which vector search misses.  
*See:* [Ch 4 › Full-text search](../ch04/README.md#full-text-search) · [Ch 4 › 1. Hybrid search: add Postgres full-text search and fuse with RRF](../ch04/README.md#1-hybrid-search-add-postgres-full-text-search-and-fuse-with-rrf)

## G

**Game day**  
A scheduled drill where you break something on purpose (here, point an alias at a provider that fails) to check that the *fallback* really works before an outage tests it for you.  
*See:* [Ch 15 › 3. Fallbacks, tested with a game day](../ch15/README.md#3-fallbacks-tested-with-a-game-day)

**Gateway (LLM gateway)**  
A reverse proxy between your apps and the model providers that speaks the OpenAI API. Apps send
ordinary requests with a gateway key and a model *alias*. The gateway adds the real provider key,
applies limits and caching, forwards the request and records the cost.  
*See:* [Ch 15 › A gateway is a reverse proxy that speaks the OpenAI API](../ch15/README.md#a-gateway-is-a-reverse-proxy-that-speaks-the-openai-api) · [Ch 15 › 1. One gateway, three aliases](../ch15/README.md#1-one-gateway-three-aliases)

**GenAI semantic conventions**  
OpenTelemetry's agreed attribute names for model calls (`gen_ai.operation.name`,
`gen_ai.usage.input_tokens`, ...). With them, any backend can read any app's traces. They are still
in *Development* status in 2026. Recording message content is opt-in, because it's the most
sensitive data you have.  
*See:* [Ch 14 › OpenTelemetry and the GenAI semantic conventions](../ch14/README.md#opentelemetry-and-the-genai-semantic-conventions) · [Ch 14 › 6. GenAI semantic conventions on the wire](../ch14/README.md#6-genai-semantic-conventions-on-the-wire)

**Golden set**  
The fixed list of eval inputs plus what we expect for each, versioned next to the code like test fixtures.  
*See:* [Ch 3 › An eval is a unit test with a score instead of a boolean](../ch03/README.md#an-eval-is-a-unit-test-with-a-score-instead-of-a-boolean) · [Ch 3 › 1. A harness and a golden set of code checks](../ch03/README.md#1-a-harness-and-a-golden-set-of-code-checks)

**`gpu_memory_utilization` (`--mem-fraction-static`)**  
The share of the *whole* GPU an engine claims at startup (vLLM's `--gpu-memory-utilization`,
SGLang's `--mem-fraction-static`). Whatever the weights don't use becomes *KV cache*, so this sets
how many requests can batch. On a shared card, set it too high and the engine refuses to start; set
it too low and it starves. `--max-model-len` caps the context per request inside that budget.  
*See:* [Ch 16 › 2. vLLM: PagedAttention and full-precision weights](../ch16/README.md#2-vllm-pagedattention-and-full-precision-weights) · [Ch 16 › 3. SGLang, and the memory knob that decided the race](../ch16/README.md#3-sglang-and-the-memory-knob-that-decided-the-race) · [Ch 17 › v0: the naive solution](../ch17/README.md#v0-the-naive-solution)

**Grammar (GBNF, XGrammar)**  
A state machine, compiled from a JSON Schema, that knows which tokens may legally come next.
*Constrained decoding* runs on it. llama.cpp (and so Ollama) uses GBNF grammars; vLLM and SGLang use
XGrammar. Engines may silently skip schema features they can't enforce.  
*See:* [Ch 2 › Constrained decoding, in one picture](../ch02/README.md#constrained-decoding-in-one-picture) · [Ch 2 › 6. Local models: same contract, different engine](../ch02/README.md#6-local-models-same-contract-different-engine)

**Grounding**  
Making the model answer from material we supply (an FAQ in the system prompt, retrieved chunks) instead of from its memory, with rules to decline when the material doesn't cover the question.  
*See:* [Ch 1 › 1. Ground the bot in the FAQ, and scope it](../ch01/README.md#1-ground-the-bot-in-the-faq-and-scope-it) · [Ch 4 › Retrieval-augmented generation](../ch04/README.md#retrieval-augmented-generation)

**Grouped-query attention (GQA)**  
An attention layout where several query heads share one key/value head. In Qwen3-8B, 32 query heads
share 8. This makes the *KV cache* smaller by the same factor (4× there). It's why KV-cache math
uses `kv_heads`, not the head count.  
*See:* [Ch 17 › Where the memory goes](../ch17/README.md#where-the-memory-goes)

**Guardrail**  
Any check *outside* the model that constrains what goes in or comes out: an input classifier, a tool allowlist, PII redaction, a URL sanitizer. (Chapter 3 also calls the scope rules inside a prompt "guardrails"; Chapter 13 shows why rules the model is merely asked to follow are not enough.)  
*See:* [Ch 13 › Guardrails are layers, and each one has a job](../ch13/README.md#guardrails-are-layers-and-each-one-has-a-job) · [Ch 3 › 2. A regression gate that can actually fail](../ch03/README.md#2-a-regression-gate-that-can-actually-fail)

## H

**Hallucination**  
A fluent, confident output that isn't supported by the input or by reality: an invented column name, a filter value that doesn't exist, a line item that isn't on the invoice. Structured outputs don't prevent it; validation and evals catch it.  
*See:* [Ch 6 › How an LLM "sees" your database](../ch06/README.md#how-an-llm-sees-your-database) · [Ch 6 › 5. Self-correction](../ch06/README.md#5-self-correction) · [Ch 8 › Field-level accuracy](../ch08/README.md#field-level-accuracy)

**Histogram (metrics)**  
A metric that counts observations per bucket (for example, requests per latency range), so percentiles can be computed after the fact with `histogram_quantile`.  
*See:* [Ch 11 › Metrics, logs and the questions each answers](../ch11/README.md#metrics-logs-and-the-questions-each-answers)

**HNSW**  
Hierarchical Navigable Small World: an approximate nearest-neighbor index. It is a multi-layer graph
of nearby vectors, and search walks it greedily, like taking motorways first and side streets last.
It is fast, but it can miss. In pgvector, a `WHERE` filter is applied after the scan.  
*See:* [Ch 4 › Embeddings and the HNSW index](../ch04/README.md#embeddings-and-the-hnsw-index) · [Ch 4 › Production hardening](../ch04/README.md#production-hardening)

**HorizontalPodAutoscaler, PodDisruptionBudget**  
Two Kubernetes objects. The HPA changes a Deployment's replica count based on a metric. The PDB
limits how many pods planned disruptions, such as node drains, may take down at once.  
*See:* [Ch 20 › Kubernetes in seven objects](../ch20/README.md#kubernetes-in-seven-objects) · [Ch 20 › 4. Autoscaling and disruption budgets](../ch20/README.md#4-autoscaling-and-disruption-budgets)

**Human-in-the-loop**  
A design where a person must approve, edit or reject an agent's action before it takes effect. In Chapter 10 the graph *pauses* with `interrupt()` and resumes when the approval arrives, possibly days later.  
*See:* [Ch 10 › 1. Pause before the side effect](../ch10/README.md#1-pause-before-the-side-effect) · [Ch 10 › 3. Validate the human, too](../ch10/README.md#3-validate-the-human-too)

**Hunk, unified diff**  
A *unified diff* is the text `git diff` prints. A *hunk* is one changed region inside it. It starts
with a header like `@@ -29,8 +32,9 @@` (old start, old length, new start, new length). Each line can
have three numbers (new line, old line, GitHub position), and bots mix them up.  
*See:* [Ch 12 › Three line numbers per line](../ch12/README.md#three-line-numbers-per-line) · [Ch 12 › 1. Hand the model line numbers instead of asking it to count](../ch12/README.md#1-hand-the-model-line-numbers-instead-of-asking-it-to-count)

**Hybrid search**  
Running vector search and full-text search side by side and merging the two rankings, usually with *RRF*. Each catches what the other misses: meaning versus exact strings.  
*See:* [Ch 4 › 1. Hybrid search: add Postgres full-text search and fuse with RRF](../ch04/README.md#1-hybrid-search-add-postgres-full-text-search-and-fuse-with-rrf)

## I

**Idempotency, idempotency key**  
An operation is idempotent if doing it twice has the same effect as doing it once. An idempotency
key identifies "the same request" (for a refund: order plus reason). A retry then replays the stored
result instead of writing a second row. Agents retry all the time, and approval alone doesn't make a
write happen exactly once.  
*See:* [Ch 9 › 4. Side effects: propose, don't perform](../ch09/README.md#4-side-effects-propose-dont-perform) · [Ch 10 › 6. Idempotent execution](../ch10/README.md#6-idempotent-execution) · [Ch 22 › D2 Retry → duplicate side effects](../ch22/README.md#d2-retry--duplicate-side-effects)

**Inference server (engine)**  
The program that loads a model's weights onto a GPU and turns HTTP requests into tokens. Examples:
vLLM, SGLang, llama.cpp's `llama-server`, and Ollama (which runs llama-server underneath). All
expose an *OpenAI-compatible API*. They differ in how they schedule requests (*continuous batching*)
and manage GPU memory (*PagedAttention*, *RadixAttention*). That difference decides throughput.  
*See:* [Ch 16 › What "serving a model" means](../ch16/README.md#what-serving-a-model-means) · [Ch 16 › 1. Turn on parallelism: Ollama with 32 slots, then llama-server itself](../ch16/README.md#1-turn-on-parallelism-ollama-with-32-slots-then-llama-server-itself)

**Input classifier**  
A small model in front of the main one. It flags harmful or abusive messages before we spend a token
on them (Llama Guard in Chapter 13). By design it can't see *indirect* prompt injection, because the
user's request itself is innocent.  
*See:* [Ch 13 › 5. An input classifier](../ch13/README.md#5-an-input-classifier) · [Ch 13 › 6. Putting it together: do we need the classifier?](../ch13/README.md#6-putting-it-together-do-we-need-the-classifier)

**`interrupt()`**  
LangGraph's call for pausing a graph in the middle of a node, saving everything, and returning a payload to the caller. Resuming *re-runs that node from the top*, so never perform a side effect above the `interrupt()`.  
*See:* [Ch 10 › interrupt(): pausing inside a node](../ch10/README.md#interrupt-pausing-inside-a-node) · [Ch 10 › Production hardening](../ch10/README.md#production-hardening) · [Ch 22 › D4 LangGraph `interrupt()` re-runs the node on resume](../ch22/README.md#d4-langgraph-interrupt-re-runs-the-node-on-resume)

## J

**Jitter**  
Adding randomness to the wait between retries, so a thousand clients that failed together don't
retry together. *Full jitter* waits a random time, picked evenly between 0 and the exponential cap.
Against a fixed rate limit, jitter spreads the retries but doesn't reduce them. A client-side
*token bucket* does.  
*See:* [Ch 21 › Retries, backoff and jitter](../ch21/README.md#retries-backoff-and-jitter) · [Ch 21 › 3. Retries: the SDK's, and when not to rely on them](../ch21/README.md#3-retries-the-sdks-and-when-not-to-rely-on-them) · [Ch 22 › F1 429 storms: synchronized retries](../ch22/README.md#f1-429-storms-synchronized-retries)

**JSON mode**  
`response_format={"type": "json_object"}`. The decoder only allows valid JSON *syntax*; keys and
values stay free. It is weaker than strict `json_schema`.  
*See:* [Ch 2 › Four ways to ask for JSON](../ch02/README.md#four-ways-to-ask-for-json)

**JSON Schema (as a contract)**  
A machine-readable description of the output's shape: field names, types, allowed values. We write
it once as a Pydantic model. From it we get the schema sent to the model, the check on the reply and
the type our code receives. It works like an OpenAPI spec between two services.  
*See:* [Ch 2 › The schema is the contract](../ch02/README.md#the-schema-is-the-contract) · [Ch 8 › Structured outputs](../ch08/README.md#structured-outputs)

**JSON-RPC**  
A small request/response protocol in JSON (method, params, id). MCP is built on it.  
*See:* [Ch 9 › MCP: tools as a service](../ch09/README.md#mcp-tools-as-a-service)

**JSONL chat format (training file)**  
The file format for fine-tuning: one JSON object per line, each a short conversation (system prompt,
user turn, assistant answer). The model never sees the JSON; a *chat template* turns each
conversation into one string. Every provider wants the same three strings in a slightly different
wrapper: OpenAI uses `messages`, Bedrock `content` blocks with a `schemaVersion`, Vertex
`contents`/`parts`. So Chapter 19 keeps one canonical example and converts and validates it per
provider.  
*See:* [Ch 18 › What the model actually trains on](../ch18/README.md#what-the-model-actually-trains-on) · [Ch 19 › 2. One canonical example, three envelopes](../ch19/README.md#2-one-canonical-example-three-envelopes) · [Ch 19 › 3. Provider-specific rules](../ch19/README.md#3-provider-specific-rules)

## K

**kind (Kubernetes in Docker)**  
A tool that runs a whole Kubernetes cluster as Docker containers on one machine. Chapter 20 uses a 3-node kind cluster to test real Kubernetes behavior (rollouts, probes, autoscaling) without a cloud account.  
*See:* [Ch 20 › On Kubernetes: kind](../ch20/README.md#on-kubernetes-kind)

**KV cache**  
The keys and values (in-between results of attention) a model keeps for every token of every active
conversation, in every layer. With them, it doesn't recompute those for the next token. Think of
it as the request's short-term memory. Its size per token is `2 × layers × kv_heads × head_dim ×
bytes` (144 KiB for Qwen3-8B in bf16). So it grows with context length × concurrent users. On a
self-hosted GPU it gets whatever memory the weights leave, and that caps how many requests can be
batched. How an engine manages it (*PagedAttention*, *RadixAttention*) largely decides throughput.
Reusing it across requests is called *prefix caching* (engine) and *prompt caching* (provider).  
*See:* [Ch 16 › The KV cache: the memory that limits the batch](../ch16/README.md#the-kv-cache-the-memory-that-limits-the-batch) · [Ch 17 › Where the memory goes](../ch17/README.md#where-the-memory-goes) · [Ch 1 › Latency: TTFT and the rest](../ch01/README.md#latency-ttft-and-the-rest)

## L

**Labels (Loki, Prometheus)**  
The few indexed key-value pairs (`service`, `level`) that identify a stream of logs or a metric
series. Each distinct combination of labels is a separate stream. So keep labels few and put
everything else in the log line.  
*See:* [Ch 11 › Metrics, logs and the questions each answers](../ch11/README.md#metrics-logs-and-the-questions-each-answers) · [Ch 11 › The telemetry plumbing](../ch11/README.md#the-telemetry-plumbing) · [Ch 22 › J2 Labeled Prometheus counter missing until first increment](../ch22/README.md#j2-labeled-prometheus-counter-missing-until-first-increment)

**LangGraph**  
A library that models an agent as a state machine. It has a typed *state*; *nodes*, which are
functions that return partial updates; and *edges*, which decide what runs next. A *conditional
edge* is a function of the state. *Reducers* merge parallel updates. *Checkpoints* are saved after
every *superstep*, the set of nodes that run together.  
*See:* [Ch 5 › LangGraph in five words: state, nodes, edges, checkpoints](../ch05/README.md#langgraph-in-five-words-state-nodes-edges-checkpoints) · [Ch 10 › Durable execution: checkpoints and threads](../ch10/README.md#durable-execution-checkpoints-and-threads)

**Lease**  
Two senses. (1) A time-limited claim on a job (`locked_until`). It is committed right away, instead
of holding a database transaction open during a slow LLM call. If the worker crashes, the lease
expires and another worker can claim the job (Chapter 21). (2) A hosted or fine-tuned model you
don't own. It works until the provider retires it, and a tuned model dies with its base model. So
keep the dataset, converter and eval, and treat the model as disposable (Chapters 19 and 22).  
*See:* [Ch 21 › 6. Async work: a Postgres queue with `FOR UPDATE SKIP LOCKED`](../ch21/README.md#6-async-work-a-postgres-queue-with-for-update-skip-locked) · [Ch 19 › A tuned model is a lease](../ch19/README.md#a-tuned-model-is-a-lease) · [Ch 22 › Model churn: your model is a lease](../ch22/README.md#model-churn-your-model-is-a-lease)

**Lethal trifecta**  
Simon Willison's framing. An agent can be exploited when it has three things together: access to
private data, exposure to untrusted content, and a way to communicate externally. You can't patch
the model's judgment. But you can remove or narrow one of the three for each request.  
*See:* [Ch 13 › The lethal trifecta](../ch13/README.md#the-lethal-trifecta)

**Little's law**  
L = λW: the average number of requests inside a system equals the arrival rate times the time each spends there. Because an LLM answer takes seconds, modest traffic means many *concurrent streams*.  
*See:* [Ch 21 › Little's law: the one formula of capacity planning](../ch21/README.md#littles-law-the-one-formula-of-capacity-planning)

**LLM judge (LLM-as-a-judge)**  
A second model that reads the question, the answer and a rubric, and returns a score. It can grade
what a regex can't. But it has biases (position, length, favoring its own answers), it costs money,
and it must be *calibrated* against humans before it gates anything.  
*See:* [Ch 3 › An eval is a unit test with a score instead of a boolean](../ch03/README.md#an-eval-is-a-unit-test-with-a-score-instead-of-a-boolean) · [Ch 3 › 3. A rubric and a reference: calibrating the judge](../ch03/README.md#3-a-rubric-and-a-reference-calibrating-the-judge)

**llm-d**  
A Kubernetes stack for serving one model on many GPUs. vLLM (or SGLang) pods sit behind a router.
The router scores each replica on queue depth, KV-cache usage and *prefix* hits, so a request goes
where its prompt is already cached. It is the step after one engine on one card; Chapter 16
describes it but didn't run it.  
*See:* [Ch 16 › Beyond one GPU: llm-d on Kubernetes](../ch16/README.md#beyond-one-gpu-llm-d-on-kubernetes)

**Logs, metrics, traces**  
The three kinds of telemetry. A *log* is one line about one event. A *metric* is a number added up
over time (requests per second, p95 latency). A *trace* is the story of one request, as a tree of
timed *spans*. LLM traces also carry prompts, tokens, cost and scores. (A business metric in a
semantic layer is a different thing: see *Metric (semantic layer)*.)  
*See:* [Ch 11 › Metrics, logs and the questions each answers](../ch11/README.md#metrics-logs-and-the-questions-each-answers) · [Ch 14 › Logs, metrics, traces, and what's different about LLMs](../ch14/README.md#logs-metrics-traces-and-whats-different-about-llms)

**LoRA, adapter**  
Low-Rank Adaptation. Instead of updating all of a model's weights, we freeze the *base model* and
train two thin matrices next to each big one. Their product is the correction. The *rank* r sets how
thin they are (r = 16 trained 1% of Qwen3-1.7B). The result, the *adapter*, is a small file (66.5 MB
in Chapter 18). It works only on the exact base model it was trained on. It can be served unmerged
next to the base, or *merged* into it. Managed providers mostly train adapters behind the scenes;
SageMaker lets you keep the weights.  
*See:* [Ch 18 › LoRA: train 1% of the weights](../ch18/README.md#lora-train-1-of-the-weights) · [Ch 18 › 1. A LoRA adapter on Qwen3-1.7B with Unsloth](../ch18/README.md#1-a-lora-adapter-on-qwen3-17b-with-unsloth) · [Ch 19 › What "managed" fine-tuning hands over](../ch19/README.md#what-managed-fine-tuning-hands-over) · [Ch 19 › SageMaker: the same LoRA, on AWS's GPU, weights in your account](../ch19/README.md#sagemaker-the-same-lora-on-awss-gpu-weights-in-your-account)

## M

**`max_tokens`**  
The cap on how many tokens the model may generate. Always set it. It limits cost and runaway
answers. Some gateways check your balance against it up front. And a reply that hits it is cut off
mid-sentence or mid-JSON (`finish_reason="length"`).  
*See:* [Ch 1 › 2. Stream the answer](../ch01/README.md#2-stream-the-answer) · [Ch 2 › 3. The truncation trap](../ch02/README.md#3-the-truncation-trap)

**MCP (Model Context Protocol)**  
A JSON-RPC protocol that offers tools (and resources and prompts) to any AI application: "USB for
tools". You write a server once, and any MCP client can find and call its tools. It runs over
*stdio* (a local subprocess) or *Streamable HTTP* (a shared service).  
*See:* [Ch 9 › MCP: tools as a service](../ch09/README.md#mcp-tools-as-a-service) · [Ch 9 › The MCP server](../ch09/README.md#the-mcp-server)

**Measure (semantic layer)**  
An aggregation over one entity, with optional row filters, such as `SUM(total_cents)` over paid orders. Metrics are built from measures.  
*See:* [Ch 7 › What a semantic layer is](../ch07/README.md#what-a-semantic-layer-is)

**Metric (semantic layer)**  
What people ask for, built from measures: *simple* (one measure), *ratio* (numerator ÷ denominator, each computed on its own) or *derived* (arithmetic over other metrics). Not to be confused with a telemetry metric.  
*See:* [Ch 7 › What a semantic layer is](../ch07/README.md#what-a-semantic-layer-is)

**MRR (mean reciprocal rank)**  
The mean of 1/rank of the first relevant result: 1.0 means it's always first, 0.5 typically second. Measures how *high* retrieval puts the right chunk, where *recall@k* only asks whether it's there.  
*See:* [Ch 4 › Retrieval-augmented generation](../ch04/README.md#retrieval-augmented-generation) · [Ch 4 › Measuring v0](../ch04/README.md#measuring-v0)

**Multi-agent**  
A design where several agents, each with its own prompts and tools, hand work to each other. Each
hand-off passes on only a summary of the context, so details get lost, and token use multiplies. So
we use one agent unless the sub-tasks are truly independent.  
*See:* [Ch 10 › One agent or many?](../ch10/README.md#one-agent-or-many)

**Multi-hop question**  
A question whose answer needs facts from different places, combined: a delivery date from SQL, a return window from the docs, then arithmetic. Neither retrieval nor text-to-SQL alone can answer it.  
*See:* [Ch 5 › Multi-hop: the answer is a join](../ch05/README.md#multi-hop-the-answer-is-a-join)

## N

**Napkin math**  
Rough estimates of cost or capacity from a few numbers: tokens per call × calls × price, or users →
peak requests per second → concurrent streams. Every assumption is printed, so nobody mistakes the
result for data.  
*See:* [Ch 1 › Cost napkin math](../ch01/README.md#cost-napkin-math) · [Ch 21 › Napkin math for a million monthly users](../ch21/README.md#napkin-math-for-a-million-monthly-users) · [Ch 16 › Prefill and decode: two different bottlenecks](../ch16/README.md#prefill-and-decode-two-different-bottlenecks) · [Ch 17 › Where the memory goes](../ch17/README.md#where-the-memory-goes)

**Nondeterminism (temperature 0)**  
Getting different outputs for the same request. Temperature 0 removes sampling, but not variation.
The provider batches your request with others, the floating-point math depends on the batch, and the
top token can flip when two are nearly tied. Plan for variation; don't assert exact strings.  
*See:* [Ch 22 › What "temperature 0" does and doesn't buy you](../ch22/README.md#what-temperature-0-does-and-doesnt-buy-you) · [Ch 22 › B2 Nondeterminism at temperature 0](../ch22/README.md#b2-nondeterminism-at-temperature-0)

## O

**Observation, generation, score (Langfuse)**  
Langfuse's names. An *observation* is a span, with typed variants. A *generation* is an observation
for one LLM call (model, usage, cost). A *score* is a number or boolean attached to a trace or
observation later, by a user, code or a judge.  
*See:* [Ch 14 › Logs, metrics, traces, and what's different about LLMs](../ch14/README.md#logs-metrics-traces-and-whats-different-about-llms) · [Ch 14 › What Langfuse adds on top](../ch14/README.md#what-langfuse-adds-on-top) · [Ch 14 › 1. One trace per question](../ch14/README.md#1-one-trace-per-question) · [Ch 14 › 4. 👍/👎 from the CLI](../ch14/README.md#4--from-the-cli)

**OCR**  
Optical character recognition: a model that transcribes the pixels of a page into text. In an *OCR then LLM* pipeline, a text model structures the transcript but never sees the page, so OCR mistakes are baked in.  
*See:* [Ch 8 › Three ways to read a page](../ch08/README.md#three-ways-to-read-a-page) · [Ch 8 › 1. Scans: vision model or OCR then LLM?](../ch08/README.md#1-scans-vision-model-or-ocr-then-llm)

**On-demand vs dedicated serving**  
The two ways to serve a model. *On-demand* (per-token) runs on the provider's shared machines. You
pay per token, and an idle model costs nothing or a small storage fee. *Dedicated* means paying for
a GPU by the hour (a SageMaker endpoint, a dedicated H100). That costs the same at 10 requests a
month as at 10 million.  
*See:* [Ch 19 › What "managed" fine-tuning hands over](../ch19/README.md#what-managed-fine-tuning-hands-over) · [Ch 19 › Training is cheap, serving decides](../ch19/README.md#training-is-cheap-serving-decides)

**OpenAI-compatible API**  
Any server that accepts the same HTTP requests as OpenAI's *Chat Completions* (Ollama, vLLM, llama.cpp, OpenRouter, LiteLLM). Changing provider is then a matter of `base_url`, `api_key` and the model name.  
*See:* [Ch 1 › The model landscape (as of 2026-11-08)](../ch01/README.md#the-model-landscape-as-of-2026-11-09) · [Ch 2 › 6. Local models: same contract, different engine](../ch02/README.md#6-local-models-same-contract-different-engine) · [Ch 15 › A gateway is a reverse proxy that speaks the OpenAI API](../ch15/README.md#a-gateway-is-a-reverse-proxy-that-speaks-the-openai-api) · [Ch 16 › What "serving a model" means](../ch16/README.md#what-serving-a-model-means)

**OpenTelemetry (OTel), OTLP**  
The vendor-neutral standard for traces, metrics and logs. It has an API to instrument your code,
SDKs that batch and export the data, and a wire protocol, OTLP. *Semantic conventions* are its
agreed attribute names.  
*See:* [Ch 14 › OpenTelemetry and the GenAI semantic conventions](../ch14/README.md#opentelemetry-and-the-genai-semantic-conventions)

**OWASP Top 10 for LLM Applications**  
The OWASP Foundation's ranked list of risks for applications built on LLMs. The 2026 edition
includes LLM01 Prompt Injection, LLM03 Excessive Agency and LLM10 Improper Output Handling, among
others. A companion list covers agentic applications. We tag every attack in Chapter 13 with its
ids.  
*See:* [Ch 13 › Mapping to the OWASP Top 10 for LLM Applications 2026](../ch13/README.md#mapping-to-the-owasp-top-10-for-llm-applications-2026)

## P

**p50, p95 (percentiles)**  
The latency below which 50% (the median) or 95% of requests finish. p95 shows what your slowest regular users feel. With only five samples, p95 is basically "the slowest run".  
*See:* [Ch 1 › Latency: TTFT and the rest](../ch01/README.md#latency-ttft-and-the-rest) · [Ch 11 › Metrics, logs and the questions each answers](../ch11/README.md#metrics-logs-and-the-questions-each-answers)

**PagedAttention**  
vLLM's way of storing the *KV cache*. It uses small fixed-size blocks handed out as tokens arrive,
like an operating system's virtual memory pages. The alternative is one big slab per request, sized
for the maximum length. Little memory is wasted, so more requests fit in a batch. Requests with the
same prefix can also share blocks, which makes *prefix caching* cheap.  
*See:* [Ch 17 › PagedAttention](../ch17/README.md#pagedattention) · [Ch 16 › The KV cache: the memory that limits the batch](../ch16/README.md#the-kv-cache-the-memory-that-limits-the-batch) · [Ch 16 › 2. vLLM: PagedAttention and full-precision weights](../ch16/README.md#2-vllm-pagedattention-and-full-precision-weights)

**Pairwise judge, position bias**  
A *pairwise* judge sees two answers and picks the better one. *Position bias* is its habit of
preferring whichever it sees first (or second), whatever the content. The fix: ask twice with the
answers swapped, and count only wins that survive the swap.  
*See:* [Ch 3 › Pairwise judging and position bias](../ch03/README.md#pairwise-judging-and-position-bias) · [Ch 3 › 5. Pairwise A/B with a position swap](../ch03/README.md#5-pairwise-ab-with-a-position-swap) · [Ch 22 › E1 LLM-judge position bias](../ch22/README.md#e1-llm-judge-position-bias)

**Parallel tool calls**  
The model asking for several tools in one turn, which saves a whole model round trip. The calls must be independent: the model can't use one's result as another's argument in the same turn.  
*See:* [Ch 9 › A tool call is a request, not an action](../ch09/README.md#a-tool-call-is-a-request-not-an-action) · [Ch 9 › 3. Parallel tool calls](../ch09/README.md#3-parallel-tool-calls)

**pass^k**  
From τ-bench: the share of tasks an agent gets right in *all* k tries. It measures reliability, where a single run measures luck.  
*See:* [Ch 10 › Trajectory evals](../ch10/README.md#trajectory-evals)

**PEFT, TRL, Unsloth**  
Python libraries for fine-tuning open models. Hugging Face's PEFT adds *LoRA* adapters to a model.
TRL runs the supervised training loop on top of it. Unsloth is a faster, lower-memory drop-in for
the same job. Chapter 18 trains the same adapters with both and compares.  
*See:* [Ch 18 › 1. A LoRA adapter on Qwen3-1.7B with Unsloth](../ch18/README.md#1-a-lora-adapter-on-qwen3-17b-with-unsloth) · [Ch 18 › 3. Unsloth vs plain PEFT/TRL](../ch18/README.md#3-unsloth-vs-plain-pefttrl)

**pgvector**  
The Postgres extension that stores embeddings and searches them by distance, with HNSW indexes. It lets RAG live in the database you already run.  
*See:* [Ch 4 › Embeddings and the HNSW index](../ch04/README.md#embeddings-and-the-hnsw-index) · [Ch 4 › Production hardening](../ch04/README.md#production-hardening)

**PII, redaction**  
PII is personally identifiable information: emails, phone numbers, card numbers, SSNs, names. *Redaction* replaces it in the model's output before anyone sees it; Chapter 13 uses regexes for the structured kinds.  
*See:* [Ch 13 › 2. Regex PII redaction](../ch13/README.md#2-regex-pii-redaction) · [Ch 6 › 4. A read-only database role](../ch06/README.md#4-a-read-only-database-role) · [Ch 14 › 3. Keep PII out of the traces](../ch14/README.md#3-keep-pii-out-of-the-traces) · [Ch 22 › H1 PII in logs](../ch22/README.md#h1-pii-in-logs)

**Planted-bug eval**  
An eval built by inserting known bugs into code and checking whether the reviewer finds them. Labels come cheap; the weakness is that planted bugs are cleaner than real ones.  
*See:* [Ch 12 › Recall, false positives, and why a planted-bug eval](../ch12/README.md#recall-false-positives-and-why-a-planted-bug-eval)

**Postmortem**  
The written account of an incident: summary, impact, timeline, causes considered with evidence, next actions. Chapter 11's agent drafts one, ranked by confidence, for a human to check.  
*See:* [Ch 11 › An agent is a loop](../ch11/README.md#an-agent-is-a-loop) · [Ch 11 › 3. A strict report schema, ranked by confidence](../ch11/README.md#3-a-strict-report-schema-ranked-by-confidence)

**Prefill, decode**  
The two phases of generating an answer. *Prefill* reads the whole prompt in one parallel pass. It is
limited by compute and makes up most of *TTFT*. *Decode* produces one token per step and reads every
weight from GPU memory each time. It is limited by memory bandwidth and sets *TPOT*. That's why long
prompts hurt time to first token, long answers hurt total time, and *batching* decode is nearly
free.  
*See:* [Ch 16 › Prefill and decode: two different bottlenecks](../ch16/README.md#prefill-and-decode-two-different-bottlenecks) · [Ch 17 › Decode is a memory-bandwidth problem](../ch17/README.md#decode-is-a-memory-bandwidth-problem) · [Ch 16 › 4. Prefill: the measurement Chapter 1 promised](../ch16/README.md#4-prefill-the-measurement-chapter-1-promised)

**Prefix caching (engine)**  
An inference server keeps the *KV cache* of earlier prompts. When a new request starts with the same
tokens, it skips that part of *prefill*. The prefix must match byte for byte from the first token.
vLLM hashes 16-token blocks and shares them across requests (on by default). llama-server reuses
each slot's previous prompt. It is the self-hosted version of provider *prompt caching*.  
*See:* [Ch 17 › Prefix caching](../ch17/README.md#prefix-caching) · [Ch 17 › 3. Prefix caching: don't prefill the same prompt twice](../ch17/README.md#3-prefix-caching-dont-prefill-the-same-prompt-twice)

**`preStop` hook, graceful shutdown**  
During a rolling deploy, Kubernetes sends the old pod `SIGTERM` *at the same time* as it removes the
pod from the Service. A `preStop` sleep keeps the pod serving while that removal spreads. Then the
server finishes its in-flight streams. All of this must fit within the grace period.  
*See:* [Ch 20 › What happens when a pod is replaced](../ch20/README.md#what-happens-when-a-pod-is-replaced) · [Ch 20 › 3. Probes and graceful shutdown](../ch20/README.md#3-probes-and-graceful-shutdown)

**Privilege separation**  
The part that can be fooled shouldn't hold the power. Give each bot the narrowest database role. And
don't let an agent act on values that only appeared in untrusted text. Chapter 13 tracks
*provenance* (where an order id came from) to enforce the second rule.  
*See:* [Ch 13 › 3. Privilege separation](../ch13/README.md#3-privilege-separation) · [Ch 9 › Who holds the privilege](../ch09/README.md#who-holds-the-privilege)

**Probes (readiness, liveness, startup)**  
HTTP checks that Kubernetes runs on each container. *Readiness* asks "should I get traffic now?" and
may check the database. *Liveness* asks "is this process stuck?" and must not depend on anything
external. *Startup* gives slow starters (a big model) time before liveness checks begin.  
*See:* [Ch 20 › Kubernetes in seven objects](../ch20/README.md#kubernetes-in-seven-objects) · [Ch 20 › 3. Probes and graceful shutdown](../ch20/README.md#3-probes-and-graceful-shutdown) · [Ch 22 › F3 Sync health check starves under load](../ch22/README.md#f3-sync-health-check-starves-under-load)

**Prometheus, PromQL**  
Prometheus scrapes metrics from services every few seconds and answers questions in its query language, PromQL, such as requests per second per status code over the last minute.  
*See:* [Ch 11 › Metrics, logs and the questions each answers](../ch11/README.md#metrics-logs-and-the-questions-each-answers) · [Ch 11 › 1. Summarize inside the tool](../ch11/README.md#1-summarize-inside-the-tool)

**Prompt caching**  
Providers keep the work done on a prompt's opening tokens. A repeat of the same *prefix* is billed
at a fraction of the input price (10% for `gpt-6-luna`, as of 2026-11). Put static content first and
variable content last. One timestamp at the top of the system prompt turns the discount off.  
*See:* [Ch 15 › Prompt caching is about prefixes](../ch15/README.md#prompt-caching-is-about-prefixes) · [Ch 1 › Serving it](../ch01/README.md#serving-it) · [Ch 10 › 5. Compact long tool outputs](../ch10/README.md#5-compact-long-tool-outputs) · [Ch 21 › Napkin math for a million monthly users](../ch21/README.md#napkin-math-for-a-million-monthly-users) · [Ch 15 › 6. Provider prompt caching](../ch15/README.md#6-provider-prompt-caching) · [Ch 22 › I1 Prompt-cache invalidation](../ch22/README.md#i1-prompt-cache-invalidation) · [Ch 17 › Prefix caching](../ch17/README.md#prefix-caching)

**Prompt injection (direct, indirect)**  
Text the model treats as an instruction when we meant it as data. It's *direct* when the person
typing writes it. It's *indirect* when it arrives inside content the model reads for someone (a
ticket, a doc, a web page). No prompt reliably prevents it, so the defenses live in code.  
*See:* [Ch 13 › Prompt injection: data that acts like code](../ch13/README.md#prompt-injection-data-that-acts-like-code) · [Ch 2 › 5. Prompt injection inside the ticket](../ch02/README.md#5-prompt-injection-inside-the-ticket) · [Ch 9 › 5. Prompt injection: tickets are data](../ch09/README.md#5-prompt-injection-tickets-are-data) · [Ch 22 › H2 Prompt injection via retrieved content](../ch22/README.md#h2-prompt-injection-via-retrieved-content)

**Prompt management**  
Storing prompts outside the code as versioned objects with labels, fetched at runtime and linked to every generation that used them, so a prompt change is traceable like a deploy.  
*See:* [Ch 14 › What Langfuse adds on top](../ch14/README.md#what-langfuse-adds-on-top) · [Ch 20 › Shipping a prompt: canary, and the gate before it](../ch20/README.md#shipping-a-prompt-canary-and-the-gate-before-it) · [Ch 14 › 5. Prompts in a registry, with labels](../ch14/README.md#5-prompts-in-a-registry-with-labels)

**Proximate cause**  
The immediate cause of an incident (the connection pool is exhausted), as opposed to the deeper one (a slow query holds the connections). How deep an agent can go depends on what its tools can see.  
*See:* [Ch 11 › Hypotheses, evidence and confidence](../ch11/README.md#hypotheses-evidence-and-confidence)

## Q

**QLoRA, NF4**  
*LoRA* with the frozen base model stored in 4 bits (the NF4 format). Each layer is expanded back as
training reaches it; the adapter itself still trains in 16 bits. It roughly halves training memory,
at some cost in speed. The adapter learned to correct a 4-bit base. So serving it on (or *merging*
it into) the 16-bit base can cost accuracy; only a field-level eval tells you.  
*See:* [Ch 18 › QLoRA: freeze the base in 4 bits](../ch18/README.md#qlora-freeze-the-base-in-4-bits) · [Ch 18 › 2. QLoRA: the same adapter in less than half the memory](../ch18/README.md#2-qlora-the-same-adapter-in-less-than-half-the-memory)

**Quantization (GGUF, AWQ, GPTQ, FP8, W8A16, MXFP4, NVFP4)**  
Storing weights in fewer bits than the usual 16-bit *bf16*. Integer formats store a scale per block
of weights: GGUF `q8_0`/`q4_K_M` for llama.cpp and Ollama, AWQ and GPTQ for vLLM. They save memory
and bandwidth, but are expanded back to 16 bits to compute. Floating-point formats (FP8, MXFP4,
NVFP4) are multiplied directly on newer GPUs. On older ones such as the RTX 3090, FP8 runs
*weight-only* (W8A16: 8-bit weights, 16-bit activations). *Decode* is limited by bandwidth, so fewer
bytes per weight is also the main speed knob. Accuracy has to be measured on your own eval.  
*See:* [Ch 17 › Number formats: bf16, FP8, INT4, MXFP4, NVFP4](../ch17/README.md#number-formats-bf16-fp8-int4-mxfp4-nvfp4) · [Ch 17 › 1. Quantization: fewer bytes per weight](../ch17/README.md#1-quantization-fewer-bytes-per-weight)

## R

**RadixAttention**  
SGLang's way of keeping the *KV cache* in a radix tree, so requests that share a prompt prefix share its cache automatically. SGLang's counterpart to vLLM's *PagedAttention* plus *prefix caching*.  
*See:* [Ch 16 › The KV cache: the memory that limits the batch](../ch16/README.md#the-kv-cache-the-memory-that-limits-the-batch) · [Ch 16 › 3. SGLang, and the memory knob that decided the race](../ch16/README.md#3-sglang-and-the-memory-knob-that-decided-the-race)

**RAG (retrieval-augmented generation)**  
Searching our own documents before asking the model and pasting the best passages into the prompt, so it answers from an open book instead of memory. Retrieval and generation fail differently, so we measure them separately.  
*See:* [Ch 4 › Retrieval-augmented generation](../ch04/README.md#retrieval-augmented-generation) · [Ch 5 › From pipeline to agent: who decides the next step?](../ch05/README.md#from-pipeline-to-agent-who-decides-the-next-step) · [Ch 18 › Prompt, RAG or fine-tune?](../ch18/README.md#prompt-rag-or-fine-tune)

**Rate limit (RPM, TPM, 429)**  
A provider's cap on requests per minute (RPM) or tokens per minute (TPM). Going over it gets an
HTTP 429. Providers enforce it continuously as a *token bucket*. So a burst can be rejected even
when the minute's total is far below the limit.  
*See:* [Ch 21 › Rate limits are token buckets](../ch21/README.md#rate-limits-are-token-buckets) · [Ch 1 › 4. Survive the network: timeouts, retries, fallbacks](../ch01/README.md#4-survive-the-network-timeouts-retries-fallbacks) · [Ch 15 › 2. A key per app, with a budget and a rate limit](../ch15/README.md#2-a-key-per-app-with-a-budget-and-a-rate-limit) · [Ch 22 › F1 429 storms: synchronized retries](../ch22/README.md#f1-429-storms-synchronized-retries)

**Read-only role**  
A database role with only `SELECT` grants, plus column grants that hide sensitive columns. The
database itself then refuses writes and hides private columns, whatever SQL the model writes. Grants
are the real boundary; `default_transaction_read_only` is only a default.  
*See:* [Ch 6 › 4. A read-only database role](../ch06/README.md#4-a-read-only-database-role) · [Ch 13 › 3. Privilege separation](../ch13/README.md#3-privilege-separation)

**Reasoning model, reasoning effort**  
Models that generate hidden "thinking" tokens before the answer. `reasoning_effort` controls how much they think, which changes latency and cost; pin it explicitly. Some reasoning models reject or silently ignore `temperature`.  
*See:* [Ch 1 › Sampling and temperature](../ch01/README.md#sampling-and-temperature) · [Ch 1 › Latency: TTFT and the rest](../ch01/README.md#latency-ttft-and-the-rest) · [Ch 9 › 6. Provider differences](../ch09/README.md#6-provider-differences) · [Ch 22 › A2 Hidden reasoning eats `max_tokens`: empty answer, still billed](../ch22/README.md#a2-hidden-reasoning-eats-max_tokens-empty-answer-still-billed) · [Ch 22 › A4 Reasoning models reject (or silently drop) `temperature`](../ch22/README.md#a4-reasoning-models-reject-or-silently-drop-temperature)

**Recall (review bot)**  
The share of real bugs the bot found: 7 of 8 is 87.5%. Read it together with false positives per PR; either alone can be gamed.  
*See:* [Ch 12 › Recall, false positives, and why a planted-bug eval](../ch12/README.md#recall-false-positives-and-why-a-planted-bug-eval)

**Recall@k**  
For retrieval: did a relevant chunk make the top k? It is measured without any LLM. Chapter 4 uses a
*doc-level* version (any chunk from the right document) and a stricter *evidence-level* one (the
chunk must contain the fact). Chapter 12 uses plain *recall* for found bugs; see *Recall (review
bot)*.  
*See:* [Ch 4 › Retrieval-augmented generation](../ch04/README.md#retrieval-augmented-generation) · [Ch 4 › Measuring v0](../ch04/README.md#measuring-v0)

**Reciprocal Rank Fusion (RRF)**  
A way to merge rankings whose scores can't be compared. Ignore the scores. Give each item 1/(k +
rank) in each list, and add them up (k = 60). An item that is decent in *both* lists beats one that
is top in only one.  
*See:* [Ch 4 › Reciprocal Rank Fusion](../ch04/README.md#reciprocal-rank-fusion) · [Ch 4 › 1. Hybrid search: add Postgres full-text search and fuse with RRF](../ch04/README.md#1-hybrid-search-add-postgres-full-text-search-and-fuse-with-rrf)

**Repair (validate, repair, escalate)**  
After a validation rule fails, a second turn in the same conversation shows the model its previous answer and the exact rule that broke, and gives it one chance to fix it. If that fails too, the case is escalated.  
*See:* [Ch 8 › 4. Validate, repair, escalate](../ch08/README.md#4-validate-repair-escalate)

**Repro**  
The smallest program that reproduces a failure. Keep it after the fix: a repro is a regression test for the outside world, and it tells you when a provider changes a default or a library fixes the bug for you.  
*See:* [Ch 22 › Repro first](../ch22/README.md#repro-first) · [Ch 22 › Running the repros](../ch22/README.md#running-the-repros)

**Responses API**  
OpenAI's newer API (`client.responses.create`) with server-side conversation state and hosted tools. We use *Chat Completions* instead because every provider implements it.  
*See:* [Ch 1 › The model landscape (as of 2026-11-08)](../ch01/README.md#the-model-landscape-as-of-2026-11-09)

**Retry, `Retry-After`**  
Sending a failed request again. Retry what a retry can fix: timeouts, 429, 5xx, and validation
errors fed back to the model. Never retry what it can't fix, such as a truncated or refused reply.
When the server sends a `Retry-After` header, the OpenAI SDK waits exactly that long. Count retries
across all layers as one budget.  
*See:* [Ch 21 › Retries, backoff and jitter](../ch21/README.md#retries-backoff-and-jitter) · [Ch 2 › 2. Retry only what a retry can fix](../ch02/README.md#2-retry-only-what-a-retry-can-fix) · [Ch 21 › Production hardening](../ch21/README.md#production-hardening)

**Roofline (memory-bandwidth bound)**  
The speed limit set by hardware, not software. For decode at batch size 1, it is memory bandwidth ÷
bytes of weights. For Qwen3-8B in bf16 on an RTX 3090: 936 GB/s ÷ 16.4 GB ≈ 57 tokens/s. Measured
numbers are best read as a percentage of that ceiling.  
*See:* [Ch 17 › Decode is a memory-bandwidth problem](../ch17/README.md#decode-is-a-memory-bandwidth-problem) · [Ch 16 › Prefill and decode: two different bottlenecks](../ch16/README.md#prefill-and-decode-two-different-bottlenecks)

**Rubric**  
The written scoring scale a grader (human or *LLM judge*) applies. Each level is described by the
failures it covers. In Chapter 3, we gave the judge the same 1–5 rubric and the FAQ as reference.
Its agreement with the chapter's reference labels rose from chance level to κ ≈ 0.8. Those labels
were written by a blind AI annotator, not a human.  
*See:* [Ch 3 › 3. A rubric and a reference: calibrating the judge](../ch03/README.md#3-a-rubric-and-a-reference-calibrating-the-judge) · [Ch 2 › 4. Does the prompt matter as much as the method?](../ch02/README.md#4-does-the-prompt-matter-as-much-as-the-method)

**Runbook**  
A short operational document for a known problem: symptoms, checks, fixes. The incident agent searches them by keyword to suggest next actions.  
*See:* [Ch 11 › 4. Keyword search over runbooks](../ch11/README.md#4-keyword-search-over-runbooks)

## S

**Sampling, temperature**  
The model gives a probability for every possible next token, and a *sampler* picks one.
*Temperature* controls how adventurous the pick is. Near 0 it almost always takes the likeliest
token: more repeatable, but not guaranteed identical. At 1 it samples in proportion to the
probabilities.  
*See:* [Ch 1 › Sampling and temperature](../ch01/README.md#sampling-and-temperature) · [Ch 22 › What "temperature 0" does and doesn't buy you](../ch22/README.md#what-temperature-0-does-and-doesnt-buy-you)

**Self-correction**  
Sending a rejected query's error (from the guard or the database) back to the model and letting it try again, up to a fixed number of attempts. Error messages written for the model are what make it work.  
*See:* [Ch 6 › 5. Self-correction](../ch06/README.md#5-self-correction) · [Ch 9 › 1. Tool errors are messages for the model](../ch09/README.md#1-tool-errors-are-messages-for-the-model) · [Ch 7 › 2. Validation errors the model can act on](../ch07/README.md#2-validation-errors-the-model-can-act-on)

**Semantic cache**  
A cache that serves a stored answer when a new question's *embedding* is within a similarity threshold of an old one. It saves whole calls, and it can be wrong: it may serve another question's answer. Compare an *exact* cache (hash of the whole request).  
*See:* [Ch 15 › Four caches, four different things](../ch15/README.md#four-caches-four-different-things) · [Ch 15 › 5. A semantic cache with pgvector](../ch15/README.md#5-a-semantic-cache-with-pgvector)

**Semantic context**  
A hand-written description of what the data *means* (money in cents, the business time zone, a business glossary, example queries) given to a text-to-SQL model instead of raw DDL.  
*See:* [Ch 6 › 1. A semantic context instead of raw DDL](../ch06/README.md#1-a-semantic-context-instead-of-raw-ddl)

**Semantic layer**  
A catalog of business metrics, written once. Every tool compiles it into SQL instead of writing its
own. Think of it as the API in front of your warehouse. It is built from entities, dimensions,
measures and metrics (dbt MetricFlow, Cube, or our own).  
*See:* [Ch 7 › What a semantic layer is](../ch07/README.md#what-a-semantic-layer-is) · [Ch 7 › Build or adopt?](../ch07/README.md#build-or-adopt)

**Semaphore**  
A counter that limits how many tasks may be inside a block at once. `asyncio.Semaphore(8)` keeps a batch job at 8 requests in flight, so we don't trip the provider's rate limit.  
*See:* [Ch 2 › Async batch with a semaphore](../ch02/README.md#async-batch-with-a-semaphore)

**Server-Sent Events (SSE)**  
A plain-HTTP way for a server to push a stream of `data:` events to the browser after one request.
It works through load balancers and proxies, as long as they don't buffer it. That's why every LLM
provider streams this way.  
*See:* [Ch 1 › Serving it](../ch01/README.md#serving-it) · [Ch 1 › Production hardening](../ch01/README.md#production-hardening)

**SFT (supervised fine-tuning)**  
*Fine-tuning* on input → expected output pairs (here, ticket → triage JSON), the kind every managed provider offers. The name contrasts it with preference or reinforcement training, which the book doesn't use.  
*See:* [Ch 18 › Fine-tuning changes the weights, not the prompt](../ch18/README.md#fine-tuning-changes-the-weights-not-the-prompt) · [Ch 18 › What the model actually trains on](../ch18/README.md#what-the-model-actually-trains-on) · [Ch 19 › What "managed" fine-tuning hands over](../ch19/README.md#what-managed-fine-tuning-hands-over) · [Ch 19 › 4. A launcher that refuses to waste money](../ch19/README.md#4-a-launcher-that-refuses-to-waste-money)

**Side effect**  
Anything a tool does beyond returning data: issuing a refund, sending a message, writing a row. Side effects are where an agent's mistakes cost money, so the model proposes them and code plus a human decide.  
*See:* [Ch 9 › 4. Side effects: propose, don't perform](../ch09/README.md#4-side-effects-propose-dont-perform) · [Ch 10 › 1. Pause before the side effect](../ch10/README.md#1-pause-before-the-side-effect)

**Silent vs loud failure**  
A *loud* failure raises (a 400, a parse error, a crash loop) and your error tracker sees it. A *silent* failure returns a plausible but wrong result: yesterday's orders, a filter that matches nothing, a truncated stream that ends cleanly. The fix is usually to make the implicit explicit and check it at the boundary.  
*See:* [Ch 22 › Silent vs loud failures](../ch22/README.md#silent-vs-loud-failures)

**`SKIP LOCKED`**  
A Postgres clause (`SELECT ... FOR UPDATE SKIP LOCKED`) that lets many workers claim different rows of a queue table without waiting on each other's locks. With it, Postgres is a perfectly good job queue at moderate scale.  
*See:* [Ch 21 › 6. Async work: a Postgres queue with `FOR UPDATE SKIP LOCKED`](../ch21/README.md#6-async-work-a-postgres-queue-with-for-update-skip-locked)

**Span**  
One timed step inside a *trace*, with a start, an end, a parent and attributes, like one frame of a call stack recorded across processes.  
*See:* [Ch 14 › Logs, metrics, traces, and what's different about LLMs](../ch14/README.md#logs-metrics-traces-and-whats-different-about-llms)

**Speculative decoding (draft model, n-gram, EAGLE-3)**  
A cheap *drafter* guesses the next few tokens. The big model checks them all in one forward pass. It
keeps the longest correct run, plus one token of its own. With greedy sampling the output is
identical. There are three kinds of drafter. A *draft model* is a small model of the same family,
with the same tokenizer. *N-gram* lookup copies spans from the prompt. An EAGLE-style head (EAGLE-3)
is trained on the target's hidden states. The speedup depends on the *acceptance rate* (the share of
drafted tokens kept) and shrinks as batches grow.  
*See:* [Ch 17 › Speculative decoding](../ch17/README.md#speculative-decoding) · [Ch 17 › 4. Speculative decoding: more than one token per weight read, for one user](../ch17/README.md#4-speculative-decoding-more-than-one-token-per-weight-read-for-one-user)

**stdio, Streamable HTTP (MCP transports)**  
The two standard ways an MCP client talks to a server. With *stdio*, the client starts the server as
a subprocess and stdout *is* the protocol, so one stray `print()` breaks it. *Streamable HTTP* is a
normal HTTP service, for shared tools behind auth and a load balancer.  
*See:* [Ch 9 › MCP: tools as a service](../ch09/README.md#mcp-tools-as-a-service) · [Ch 9 › The agent, through an MCP client](../ch09/README.md#the-agent-through-an-mcp-client)

**Step cap, token budget (agent run)**  
Hard stops for an agent loop: at most N model calls, and at most N tokens per run. They are checked
*before* running the tools the model just asked for. Side effects that never ran are better than
side effects without limit.  
*See:* [Ch 9 › 2. Hard stops: step cap and token budget](../ch09/README.md#2-hard-stops-step-cap-and-token-budget) · [Ch 13 › 4. A tool allowlist per role, and a call budget](../ch13/README.md#4-a-tool-allowlist-per-role-and-a-call-budget) · [Ch 22 › D3 Agent cost runaway loop](../ch22/README.md#d3-agent-cost-runaway-loop)

**Streaming**  
Receiving the answer in chunks as it's generated (`stream=True`) instead of all at once, so the user sees text after the *TTFT* rather than after the full answer.  
*See:* [Ch 1 › 2. Stream the answer](../ch01/README.md#2-stream-the-answer) · [Ch 20 › What happens when a pod is replaced](../ch20/README.md#what-happens-when-a-pod-is-replaced) · [Ch 22 › F2 Streaming disconnects behind proxies](../ch22/README.md#f2-streaming-disconnects-behind-proxies)

**Strict mode (`json_schema`, strict tool call)**  
Asking the provider to constrain generation to our JSON Schema (`"strict": True`), so the output always parses and always has every field. It doesn't make the values right, and each provider supports a different subset of JSON Schema.  
*See:* [Ch 2 › Four ways to ask for JSON](../ch02/README.md#four-ways-to-ask-for-json) · [Ch 2 › 1. Strict `json_schema` as the default](../ch02/README.md#1-strict-json_schema-as-the-default) · [Ch 8 › Structured outputs](../ch08/README.md#structured-outputs)

**Structured outputs**  
Getting a typed object instead of prose from the model, enforced by a schema: the umbrella for JSON mode, strict `json_schema` and forced tool calls.  
*See:* [Ch 2 › Four ways to ask for JSON](../ch02/README.md#four-ways-to-ask-for-json) · [Ch 8 › Structured outputs](../ch08/README.md#structured-outputs) · [Ch 6 › 2. A structured plan instead of free text](../ch06/README.md#2-a-structured-plan-instead-of-free-text)

**Synthetic data**  
Training examples generated by a model rather than collected. Chapter 18 generated 290 messy tickets
(sarcasm, other languages, injections) to cover what the real training set lacked. It kept only
those an independent pass agreed on. It also checked them for overlap with the test set, so they
couldn't leak the exam.  
*See:* [Ch 18 › 5. v2: add 290 synthetic messy tickets](../ch18/README.md#5-v2-add-290-synthetic-messy-tickets)

**System prompt**  
The `system` message at the top of a conversation that sets the rules: scope, tone, what to refuse, and often reference material such as the FAQ. It's resent on every call, so it dominates input tokens and is the best *prompt caching* candidate.  
*See:* [Ch 1 › An LLM API is a function from text to text, billed in tokens](../ch01/README.md#an-llm-api-is-a-function-from-text-to-text-billed-in-tokens) · [Ch 1 › 1. Ground the bot in the FAQ, and scope it](../ch01/README.md#1-ground-the-bot-in-the-faq-and-scope-it)

## T

**Teacher model (teacher labels)**  
A stronger model used to label training data for a smaller one. In Chapter 18, `gpt-6-luna` wrote
the `sentiment` and `summary` fields, while the routing labels stayed gold. Where teacher and gold
disagree on whole templates, that's a policy difference, not noise.  
*See:* [Ch 18 › Half the target comes from a teacher](../ch18/README.md#half-the-target-comes-from-a-teacher)

**Tensor parallelism, pipeline parallelism**  
Two ways to split a model across GPUs. *Tensor parallelism* cuts every layer's matrices across GPUs.
Each token gets faster, but the GPUs exchange results twice per layer and need a fast link (NVLink).
*Pipeline parallelism* gives each GPU a range of layers. It needs little traffic and works across
nodes, but a request visits the stages one after another. Rule of thumb: use the smallest
tensor-parallel size that fits in one node, and pipeline only across nodes. Described, not run.  
*See:* [Ch 17 › When one GPU isn't enough: tensor and pipeline parallelism](../ch17/README.md#when-one-gpu-isnt-enough-tensor-and-pipeline-parallelism)

**Text layer**  
The characters a born-digital PDF contains, with their positions. Text extraction reads them exactly, but loses structure such as tables; scans and photos have no text layer, only pixels.  
*See:* [Ch 8 › Three ways to read a page](../ch08/README.md#three-ways-to-read-a-page)

**Thread (LangGraph)**  
One conversation, identified by a `thread_id` you choose, pointing at its latest *checkpoint*. Resuming with the same `thread_id` continues where it stopped. (Not an OS thread, and not a Slack thread.)  
*See:* [Ch 10 › Durable execution: checkpoints and threads](../ch10/README.md#durable-execution-checkpoints-and-threads) · [Ch 10 › Production hardening](../ch10/README.md#production-hardening)

**Three-way match**  
The accounts-payable check that an invoice matches a purchase order and a goods-received record, against your vendor master. Extraction isn't authorization; this is the step that is.  
*See:* [Ch 8 › Production hardening](../ch08/README.md#production-hardening)

**Throughput (aggregate tokens/s)**  
Total output tokens per second across all the requests a server is handling. Batching gives up a
little per-request *TPOT* for a lot of throughput. Throughput per GPU-hour sets the cost per token
of self-hosting.  
*See:* [Ch 16 › What we measure](../ch16/README.md#what-we-measure) · [Ch 16 › Prefill and decode: two different bottlenecks](../ch16/README.md#prefill-and-decode-two-different-bottlenecks) · [Ch 17 › What a million tokens costs](../ch17/README.md#what-a-million-tokens-costs)

**Time grain**  
The bucket size of a time series: day, week, month. Buckets and period boundaries must be computed in the *business time zone*, and periods such as "last week" are resolved by code, not by the model.  
*See:* [Ch 7 › Time grains and the business time zone](../ch07/README.md#time-grains-and-the-business-time-zone) · [Ch 7 › 3. Time ranges as named periods, resolved by code](../ch07/README.md#3-time-ranges-as-named-periods-resolved-by-code)

**Timeout**  
The longest we wait for a response. The OpenAI SDK's default is 10 minutes, far too long for a chat.
It also applies to each phase separately, so treat it as rough and add a deadline on the first
token.  
*See:* [Ch 1 › 4. Survive the network: timeouts, retries, fallbacks](../ch01/README.md#4-survive-the-network-timeouts-retries-fallbacks) · [Ch 21 › Production hardening](../ch21/README.md#production-hardening)

**Token**  
The unit a model reads, writes and bills in: a short word, part of a long word, a punctuation mark.
Tokenizers differ. Non-English text and emoji usually cost more tokens per character. Your local
count is an estimate; the provider's `usage` is the bill. (The tokens in a *token bucket* are
rate-limit permits, not text.)  
*See:* [Ch 1 › An LLM API is a function from text to text, billed in tokens](../ch01/README.md#an-llm-api-is-a-function-from-text-to-text-billed-in-tokens) · [Ch 1 › Tokens are not words, and not the same everywhere](../ch01/README.md#tokens-are-not-words-and-not-the-same-everywhere)

**Token bucket**  
A rate-limiting algorithm. A bucket holds up to *capacity* tokens and refills at a steady *rate*.
Each request spends one token (or, for a TPM limit, one per LLM token). An empty bucket means 429.
Running one on our side, shared in Redis, keeps us under the provider's limit.  
*See:* [Ch 21 › Rate limits are token buckets](../ch21/README.md#rate-limits-are-token-buckets) · [Ch 21 › 4. A Redis token bucket shared by every replica](../ch21/README.md#4-a-redis-token-bucket-shared-by-every-replica)

**Token budget (context)**  
Our own cap on how many tokens a conversation may send per request, far below the context window. When history exceeds it, we drop the oldest complete turns.  
*See:* [Ch 1 › Context window, and why we set a smaller budget](../ch01/README.md#context-window-and-why-we-set-a-smaller-budget) · [Ch 1 › 3. Remember the conversation, within a budget](../ch01/README.md#3-remember-the-conversation-within-a-budget)

**Tool call (function calling)**  
The model's way to ask us to run a function. Instead of text, it returns `tool_calls` with a name
and JSON arguments (a string that may be invalid). *We* run the function and send the result back as
a `tool` message. The model never runs anything itself.  
*See:* [Ch 9 › A tool call is a request, not an action](../ch09/README.md#a-tool-call-is-a-request-not-an-action) · [Ch 2 › Four ways to ask for JSON](../ch02/README.md#four-ways-to-ask-for-json)

**Tool loop (plain)**  
The simplest *agent*. A `while` loop calls the model and runs any *tool calls* it asks for. It
appends the results and stops when the model answers or a *step cap* is hit. It is about 40 lines.
It's the right choice until you need *checkpoints* or explicit control flow (*LangGraph*).  
*See:* [Ch 5 › 1. Give the model tools: a plain tool loop](../ch05/README.md#1-give-the-model-tools-a-plain-tool-loop)

**Trajectory eval**  
Grading everything an agent *did* (each tool call and its arguments, each approval request, each write, and the final database state) rather than just its final message. A polite "I've escalated this" means nothing if a refund row was written anyway.  
*See:* [Ch 10 › Trajectory evals](../ch10/README.md#trajectory-evals)

**TTFT, TPOT, total latency**  
*Time to first token* is how long the user stares at nothing. *Time per output token* is the steady
rate after that. *Total latency* is when the answer is complete. Users judge speed by TTFT, and
every queue in the request path adds to it. On a self-hosted engine, TTFT is mostly *prefill* (plus
waiting for a batch slot), and TPOT is *decode*.  
*See:* [Ch 1 › Latency: TTFT and the rest](../ch01/README.md#latency-ttft-and-the-rest) · [Ch 21 › Three queues inside one Python process](../ch21/README.md#three-queues-inside-one-python-process) · [Ch 16 › Prefill and decode: two different bottlenecks](../ch16/README.md#prefill-and-decode-two-different-bottlenecks) · [Ch 16 › What we measure](../ch16/README.md#what-we-measure)

**TTL (time to live)**  
How long a cached entry stays valid before it expires. An exact cache can only be wrong if the world changed within its TTL; provider prompt caches last minutes.  
*See:* [Ch 15 › Four caches, four different things](../ch15/README.md#four-caches-four-different-things) · [Ch 15 › Prompt caching is about prefixes](../ch15/README.md#prompt-caching-is-about-prefixes)

## V

**Validation rules**  
Checks on an extraction that don't depend on the model. They use the redundancy in the data: line
items must add up to the subtotal, and subtotal plus tax must equal the total. They can't prove an
extraction right, but they prove many wrong. The dangerous case is *silently wrong*.  
*See:* [Ch 8 › Invoices check themselves](../ch08/README.md#invoices-check-themselves) · [Ch 8 › 4. Validate, repair, escalate](../ch08/README.md#4-validate-repair-escalate)

**Verifier**  
A second, cheap model call that checks a draft answer against the source (the FAQ) and decides whether a *cascade* must escalate to the expensive model.  
*See:* [Ch 15 › A cascade spends the strong model only where it matters](../ch15/README.md#a-cascade-spends-the-strong-model-only-where-it-matters)

**Virtual key**  
A key issued by the *gateway* to one app, mapped to the real provider key that only the gateway holds. Each virtual key carries its own budget, *rate limit* and allowed models, and revoking one app is one API call.  
*See:* [Ch 15 › 2. A key per app, with a budget and a rate limit](../ch15/README.md#2-a-key-per-app-with-a-budget-and-a-rate-limit)

**Vision LLM (multimodal model)**  
A model that reads images. The page is cut into small square patches, each patch becomes a few
tokens, and the model reads them like words. It reads and structures in one step. How many pixels
you send changes both cost and accuracy.  
*See:* [Ch 8 › Three ways to read a page](../ch08/README.md#three-ways-to-read-a-page) · [Ch 8 › 2. How many pixels does a page need?](../ch08/README.md#2-how-many-pixels-does-a-page-need)

## W

**Weights (parameters), bf16**  
The numbers that make up a model. A model on disk is a few files of them (`safetensors`). Memory
needed ≈ parameters × bytes per parameter. Qwen3-8B's 8.19 billion parameters in *bf16* (a 16-bit
float, 2 bytes) take 15.3 GiB before any *KV cache*. *Quantization* shrinks the bytes per
parameter.  
*See:* [Ch 17 › Where the memory goes](../ch17/README.md#where-the-memory-goes) · [Ch 16 › What "serving a model" means](../ch16/README.md#what-serving-a-model-means)

**Worker-thread pool**  
The pool of threads (40 by default in AnyIO) where FastAPI runs plain `def` endpoints and sync
generators, so they don't block the event loop. It silently caps concurrency. When it's full,
requests queue and TTFT grows.  
*See:* [Ch 21 › Three queues inside one Python process](../ch21/README.md#three-queues-inside-one-python-process) · [Ch 1 › Serving it](../ch01/README.md#serving-it)

**Workflow**  
Code that calls an LLM at fixed steps (classify, then look up, then draft). It is cheaper, faster
and easier to test than an *agent*. Use it when the path doesn't branch. Also use it for the step
with real consequences, even inside an agent.  
*See:* [Ch 5 › From pipeline to agent: who decides the next step?](../ch05/README.md#from-pipeline-to-agent-who-decides-the-next-step) · [Ch 10 › Workflow or agent?](../ch10/README.md#workflow-or-agent)

<!--
Glossary maintenance notes (for the integration owner)

Scanned on 2026-11-09, 192 terms.
Finished chapters scanned in full: ch01-ch22 (all chapters).

No pending terms. Topics deliberately folded into other entries rather than given their own:
mixture of experts, CUDA graphs, Marlin kernels, goodput/open-loop (in Closed-loop benchmark),
adapter merging and LoRA rank (in LoRA, adapter / QLoRA), test segments (in Data leakage).

Link check: every relative link + anchor in this file was resolved against the GitHub heading slugs
of the target README (fenced code skipped, duplicate slugs suffixed) with a one-off script (not
committed).
-->
