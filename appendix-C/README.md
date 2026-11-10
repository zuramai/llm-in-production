# Appendix C: References & further reading

Throughout the book, chapters cite the docs, papers and incidents they build on. Use this appendix
when you want to go deeper on a topic, check a fact for yourself, or find which chapter applies
a given paper or doc.

Every chapter ends with a `## References` list. This appendix merges them into one place: 535
citations from 22 chapters and [Appendix A](../appendix-A/README.md), deduplicated into **487
sources**, grouped by topic. Each entry says which chapters cite it. So you can jump from a
source to the place where the book puts it to work.

## How these sources were chosen and checked

- **Chosen while writing, not afterwards.** Before writing a chapter, its author looked up the current versions and APIs of the tools in the official docs, at least three verifiable "in the wild" facts (company case studies, incidents, postmortems) and at least three practitioner pain points from Hacker News, Stack Overflow, GitHub issues or forums ([style guide](../docs/style-guide.md#sources-and-research)).
- **Checked before citing.** Each URL was opened and the claim found on the page. Sources that couldn't be verified were dropped. Chapters quote at most two sentences from any source and paraphrase the rest.
- **Nothing new here.** Every entry below comes from a chapter's `## References`; this appendix adds no source and no claim.
- **Duplicates.** Two citations are one source when their URLs match after ignoring the scheme, the host's letter case, a trailing slash and any `#fragment`. One source, Anthropic's prompt-caching docs, is cited under an old URL in Chapter 17 and the current one in Chapters 15 and 22; the old URL redirects to the new one, so they are merged and both URLs are shown. Different articles about the same story (the $1 Chevy chatbot, the Replit database deletion) stay separate.

### Dates

An entry carries the date the source was published when it has one. Living pages (API docs, READMEs, model cards, pricing pages, package indexes) carry the date the author last opened them instead: "accessed 2026-11-07". When two chapters checked the same page on different days, the entry shows the earlier day. All sources were checked between 2026-11-07 and 2026-11-09, so prices, version numbers and model names are as of 2026-11.

### When a link breaks

- **Docs move more often than they disappear.** Search the title on the publisher's site first. Old URLs often redirect: Anthropic's `docs.anthropic.com/en/docs/build-with-claude/prompt-caching`, cited in Chapter 17, now lands on `platform.claude.com`. Redirects don't last forever, though.
- **Use the Wayback Machine for articles and press releases:** `https://web.archive.org/web/2026*/<url>`. Two entries already point to an archived copy because the original is gone (the U.S. Attorney's press release in Chapter 8 and Aim Labs' EchoLeak write-up in Chapter 13).
- **GitHub issues survive repository renames** (GitHub redirects), but not deleted repositories. Read the issue's state too: an issue a chapter cites as open may be fixed by the time you read it.
- **"Accessed" pages change under the same URL.** A pricing table or a model page may say something different today. The chapter text records what it said on the accessed date; trust the live page for decisions and the chapter for the reasoning.

## Start here

If you only read a few sources, read these. They are the thirteen sources the book leans on
most: each is cited by several chapters, or is the reference a whole section is built around.
One sentence each on how the book uses it.

1. **[Air Canada chatbot ruling (CBC)](https://www.cbc.ca/news/canada/british-columbia/air-canada-chatbot-lawsuit-1.7116416)** ([Ch 1](../ch01/README.md#references), [Ch 22](../ch22/README.md#references)). A tribunal ordered the airline to pay for its website chatbot's wrong advice, ruling that a company is responsible for everything on its website, chatbot included; the book opens with it.
2. **[OpenAI: Structured model outputs](https://developers.openai.com/api/docs/guides/structured-outputs)** ([Ch 2](../ch02/README.md#references), [Ch 8](../ch08/README.md#references), [Ch 22 › A1](../ch22/README.md#a1-finish_reason--length-truncates-json), [Ch 22 › B3](../ch22/README.md#b3-safety-filter-empty-reply-finish_reasoncontent_filter)). The rules of strict JSON mode (every field `required`, `additionalProperties: false`, optional fields as nullable) and the reminder to check `finish_reason` before parsing.
3. **[Zheng et al.: Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)** ([Ch 3](../ch03/README.md#references), [Ch 22 › E1](../ch22/README.md#e1-llm-judge-position-bias)). The paper behind LLM-as-a-judge: GPT-4 agreed with human preferences over 80% of the time, and it names the position, verbosity and self-enhancement biases that Chapter 3 measures.
4. **[Anthropic: Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)** ([Ch 4](../ch04/README.md#references)). Adding BM25 to embeddings and prepending chunk context cut failed retrievals by 49%, and by 67% with reranking: the case for hybrid search in Chapter 4.
5. **[pgvector README](https://github.com/pgvector/pgvector)** ([Ch 4](../ch04/README.md#references), [Ch 15](../ch15/README.md#references), [Ch 22 › C1](../ch22/README.md#c1-filtered-hnsw-query-returns-fewer-rows-than-limit)). Filtering and iterative index scans: why a filtered HNSW query can return fewer rows than `LIMIT`, and the setting that fixes it.
6. **[Anthropic: Building effective agents](https://www.anthropic.com/research/building-effective-agents)** ([Ch 5](../ch05/README.md#references), [Ch 10](../ch10/README.md#references)). Where the line between a workflow and an agent sits, and the advice to start with the simplest thing that works.
7. **[Invariant Labs: GitHub MCP exploited](https://invariantlabs.ai/blog/mcp-github-vulnerability)** ([Ch 9](../ch09/README.md#references), [Ch 13](../ch13/README.md#references), [Ch 22 › H2](../ch22/README.md#h2-prompt-injection-via-retrieved-content)). A public GitHub issue steered an agent with the official GitHub MCP server into leaking private repositories, a flaw of the agent system rather than of the server code.
8. **[Simon Willison: The lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)** ([Ch 13](../ch13/README.md#references), [Ch 22 › H2](../ch22/README.md#h2-prompt-injection-via-retrieved-content)). Private data, untrusted content and a way to send data out: an agent with all three is exploitable. Chapter 13 maps ShopFlow's support assistant onto it.
9. **[Anthropic: A postmortem of three recent issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues)** ([Ch 14](../ch14/README.md#references), [Ch 20](../ch20/README.md#references), [Ch 22 › J1](../ch22/README.md#j1-model-deprecation-and-silent-drift)). Three infrastructure bugs degraded Claude's answers for weeks while the evals stayed green; the fixes include running quality evals continuously on production.
10. **[OpenAI: Expanding on what we missed with sycophancy](https://openai.com/index/expanding-on-sycophancy/)** ([Ch 14](../ch14/README.md#references), [Ch 20](../ch20/README.md#references)). A GPT-4o update passed offline evals and small A/B tests and still had to be rolled back for sycophancy; OpenAI concluded that model behavior issues must block a launch.
11. **[Kwon et al.: PagedAttention](https://arxiv.org/abs/2309.06180)** ([Ch 17](../ch17/README.md#references)). The paper behind vLLM: managing the KV cache in pages gave 2–4× the throughput of FasterTransformer and Orca at the same latency.
12. **[Biderman et al.: LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)** ([Ch 18](../ch18/README.md#references)). LoRA underperforms full fine-tuning on new skills but better preserves the base model's behavior outside the target domain, which fits Chapter 18's hard-ticket numbers.
13. **[AWS: Exponential Backoff And Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)** ([Ch 21](../ch21/README.md#references), [Ch 22 › F1](../ch22/README.md#f1-429-storms-synchronized-retries)). The reference for retries: add jitter so that clients that failed together don't retry together.

## Contents

| Topic | Sources |
|---|---:|
| [Model APIs & providers](#model-apis--providers) | 13 |
| [Structured outputs](#structured-outputs) | 17 |
| [Evaluation & benchmarks](#evaluation--benchmarks) | 15 |
| [Retrieval & RAG](#retrieval--rag) | 10 |
| [Text-to-SQL & semantic layers](#text-to-sql--semantic-layers) | 26 |
| [Documents & multimodal input](#documents--multimodal-input) | 13 |
| [Agents, tools & MCP](#agents-tools--mcp) | 25 |
| [Security & guardrails](#security--guardrails) | 28 |
| [Observability & incident analysis](#observability--incident-analysis) | 23 |
| [Gateways, caching & cost](#gateways-caching--cost) | 14 |
| [Serving & GPUs](#serving--gpus) | 49 |
| [Fine-tuning](#fine-tuning) | 41 |
| [Code-review bots & the GitHub API](#code-review-bots--the-github-api) | 6 |
| [Deployment & scaling](#deployment--scaling) | 39 |
| [Dates, money & locales](#dates-money--locales) | 7 |
| [Incidents & postmortems](#incidents--postmortems) | 12 |
| [Regulation](#regulation) | 4 |
| [Practitioner threads](#practitioner-threads) | 145 |
| **Total** | **487** |

Each entry reads: title — publisher or author, date. URL — the chapters that cite it. Links to Chapter 22 go straight to the edge case (A1 to J2) that uses the source.

## Model APIs & providers

- "Deprecations" (including "Update to OpenAI's self-serve fine-tuning", 2026-05-07, updated 2026-10-14) — OpenAI docs, accessed 2026-11-08. <https://developers.openai.com/api/docs/deprecations> — cited in [Ch 19](../ch19/README.md#references), [Ch 22 › J1](../ch22/README.md#j1-model-deprecation-and-silent-drift)
- "Gemini deprecations" — Google AI Studio docs, accessed 2026-11-08. <https://aistudio.google.com/docs/deprecations> — cited in [Ch 22 › J1](../ch22/README.md#j1-model-deprecation-and-silent-drift)
- "GPT-6 Luna" model page — OpenAI API docs, accessed 2026-11-07. <https://developers.openai.com/api/docs/models/gpt-6-luna> — cited in [Ch 1](../ch01/README.md#references), [Ch 8](../ch08/README.md#references), [Ch 9](../ch09/README.md#references), [App. A](../appendix-A/README.md#references)
- Chen, Zaharia, Zou, "How is ChatGPT's behavior changing over time?" — arXiv, 2023. <https://arxiv.org/abs/2307.09009> — cited in [Ch 22 › J1](../ch22/README.md#j1-model-deprecation-and-silent-drift)
- "How to stream completions" — OpenAI Cookbook. <https://developers.openai.com/cookbook/examples/how_to_stream_completions> — cited in [Ch 1](../ch01/README.md#references)
- "Model deprecations" — Anthropic docs, accessed 2026-11-08. <https://platform.claude.com/docs/en/about-claude/model-deprecations> — cited in [Ch 22 › J1](../ch22/README.md#j1-model-deprecation-and-silent-drift)
- "OpenAI launches GPT-6 Sol and Luna" — TechCrunch, 2026-09-22. <https://techcrunch.com/2026/09/22/openai-launches-gpt-6-sol-and-luna/> — cited in [Ch 1](../ch01/README.md#references)
- "OpenAI SDK compatibility" — Anthropic docs, accessed 2026-11-08. <https://platform.claude.com/docs/en/api/openai-sdk> — cited in [Ch 22 › A4](../ch22/README.md#a4-reasoning-models-reject-or-silently-drop-temperature), [Ch 22 › B1](../ch22/README.md#b1-json_object-returns-fenced-json)
- openai-python 3.46.0 — PyPI, 2026-11-05. <https://pypi.org/project/openai/> — cited in [Ch 19](../ch19/README.md#references)
- OpenRouter models API (prices) — OpenRouter, accessed 2026-11-07. <https://openrouter.ai/api/v1/models> — cited in [Ch 1](../ch01/README.md#references), [Ch 17](../ch17/README.md#references), [App. A](../appendix-A/README.md#references)
- "Pricing" — OpenAI, accessed 2026-11-08. <https://developers.openai.com/api/docs/pricing> — cited in [Ch 19](../ch19/README.md#references)
- "Provider Routing: requiring providers to support all parameters" — OpenRouter docs, accessed 2026-11-08. <https://openrouter.ai/docs/guides/routing/provider-selection> — cited in [Ch 22 › A4](../ch22/README.md#a4-reasoning-models-reject-or-silently-drop-temperature), [App. A](../appendix-A/README.md#references)
- "Reasoning models", section "Allocating space for reasoning" — OpenAI docs, accessed 2026-11-08. <https://developers.openai.com/api/docs/guides/reasoning> — cited in [Ch 22 › A2](../ch22/README.md#a2-hidden-reasoning-eats-max_tokens-empty-answer-still-billed)

## Structured outputs

- "AI-Powered Engineering Support" — Remote Engineering Blog, 2026-03-13. <https://engineering.remote.com/blog/ai-powered-engineering-support> — cited in [Ch 2](../ch02/README.md#references)
- GBNF Guide — llama.cpp, accessed 2026-11-07. <https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md> — cited in [Ch 2](../ch02/README.md#references)
- Instructor — 567 Labs, v1.17.0, accessed 2026-11-07. <https://github.com/567-labs/instructor> — cited in [Ch 2](../ch02/README.md#references)
- "Introducing Structured Outputs in the API" — OpenAI, 2024-08-06. <https://openai.com/index/introducing-structured-outputs-in-the-api/> — cited in [Ch 2](../ch02/README.md#references)
- Geng et al., "JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models" — arXiv, 2025-01-18. <https://arxiv.org/abs/2501.10868> — cited in [Ch 2](../ch02/README.md#references)
- Tam et al., "Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models" — arXiv, 2024-08-05. <https://arxiv.org/abs/2408.02442> — cited in [Ch 2](../ch02/README.md#references)
- Outlines — dottxt, v1.3.3, accessed 2026-11-07. <https://github.com/dottxt-ai/outlines> — cited in [Ch 2](../ch02/README.md#references)
- Will Kurt, "Say What You Mean: A Response to 'Let Me Speak Freely'" — .txt blog. <https://blog.dottxt.ai/say-what-you-mean.html> — cited in [Ch 2](../ch02/README.md#references)
- Strict tool use — Anthropic docs, accessed 2026-11-07. <https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use> — cited in [Ch 2](../ch02/README.md#references)
- "Structured model outputs" — OpenAI docs, accessed 2026-11-07. <https://developers.openai.com/api/docs/guides/structured-outputs> — cited in [Ch 2](../ch02/README.md#references), [Ch 8](../ch08/README.md#references), [Ch 22 › A1](../ch22/README.md#a1-finish_reason--length-truncates-json), [Ch 22 › B3](../ch22/README.md#b3-safety-filter-empty-reply-finish_reasoncontent_filter)
- Structured outputs — Anthropic (Claude Platform) docs, accessed 2026-11-07. <https://platform.claude.com/docs/en/build-with-claude/structured-outputs> — cited in [Ch 2](../ch02/README.md#references)
- Structured Outputs — OpenRouter docs, accessed 2026-11-07. <https://openrouter.ai/docs/guides/features/structured-outputs> — cited in [Ch 2](../ch02/README.md#references)
- Structured Outputs — Ollama docs, accessed 2026-11-07. <https://docs.ollama.com/capabilities/structured-outputs> — cited in [Ch 2](../ch02/README.md#references)
- Structured Outputs — vLLM docs, accessed 2026-11-07. <https://docs.vllm.ai/en/latest/features/structured_outputs/> — cited in [Ch 2](../ch02/README.md#references)
- Ticket routing guide — Anthropic docs, accessed 2026-11-07. <https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing> — cited in [Ch 2](../ch02/README.md#references)
- "TickIt: Leveraging Large Language Models for Automated Ticket Escalation" — ByteDance, arXiv, 2025-04-11. <https://arxiv.org/abs/2504.08475> — cited in [Ch 2](../ch02/README.md#references)
- Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" — arXiv, 2024-11-22. <https://arxiv.org/abs/2411.15100> — cited in [Ch 2](../ch02/README.md#references)

## Evaluation & benchmarks

- "A statistical approach to model evaluations" — Anthropic, 2024-11-19. <https://www.anthropic.com/research/statistical-approach-to-model-evals> — cited in [Ch 22 › E3](../ch22/README.md#e3-small-eval-gates-are-noisy)
- Xiong et al., "Can LLMs Express Their Uncertainty?" — ICLR 2024. <https://arxiv.org/abs/2306.13063> — cited in [Ch 22 › E2](../ch22/README.md#e2-self-reported-confidence-is-uncalibrated)
- deepeval 4.3.3 — PyPI, 2026-11-04. <https://pypi.org/project/deepeval/4.3.3/> — cited in [Ch 3](../ch03/README.md#references)
- Bowyer, Aitchison, Ivanova, "Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints" — arXiv, 2025. <https://arxiv.org/abs/2503.01747> — cited in [Ch 22 › E3](../ch22/README.md#e3-small-eval-gates-are-noisy)
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" — NeurIPS 2023 Datasets and Benchmarks, arXiv 2306.05685. <https://arxiv.org/abs/2306.05685> — cited in [Ch 3](../ch03/README.md#references), [Ch 22 › E1](../ch22/README.md#e1-llm-judge-position-bias)
- Shi et al., "Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge" — arXiv, 2024. <https://arxiv.org/abs/2406.07791> — cited in [Ch 22 › E1](../ch22/README.md#e1-llm-judge-position-bias)
- "Meta's vanilla Maverick AI model ranks below rivals on a popular chat benchmark" — TechCrunch, 2025-04-11. <https://techcrunch.com/2025/04/11/metas-vanilla-maverick-ai-model-ranks-below-rivals-on-a-popular-chat-benchmark/> — cited in [Ch 3](../ch03/README.md#references)
- nodejs-wheel (Node.js binaries on PyPI) — PyPI, accessed 2026-11-07. <https://pypi.org/project/nodejs-wheel/> — cited in [Ch 3](../ch03/README.md#references)
- "OpenAI acquires Promptfoo to secure its AI agents" — TechCrunch, 2026-03-09. <https://techcrunch.com/2026/03/09/openai-acquires-promptfoo-to-secure-its-ai-agents/> — cited in [Ch 3](../ch03/README.md#references)
- promptfoo 0.125.1 release — GitHub, 2026-10-25. <https://github.com/promptfoo/promptfoo/releases/tag/0.125.1> — cited in [Ch 3](../ch03/README.md#references)
- promptfoo package metadata (`engines.node >= 22.22.0`) — npm registry, accessed 2026-11-07. <https://registry.npmjs.org/promptfoo> — cited in [Ch 3](../ch03/README.md#references)
- ragas 0.4.3 — PyPI, 2026-01-13. <https://pypi.org/project/ragas/0.4.3/> — cited in [Ch 3](../ch03/README.md#references)
- `sklearn.metrics.cohen_kappa_score` — scikit-learn 1.9 documentation, accessed 2026-11-07. <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html> — cited in [Ch 3](../ch03/README.md#references)
- "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains" — Yao et al., arXiv 2406.12045, 2024. <https://arxiv.org/abs/2406.12045> — cited in [Ch 10](../ch10/README.md#references)
- "τ-bench: Benchmarking AI agents for the real world" — Sierra, 2024. <https://sierra.ai/blog/benchmarking-ai-agents> — cited in [Ch 10](../ch10/README.md#references)

## Retrieval & RAG

- "Contextual Retrieval in AI Systems" — Anthropic, 2024-09-19. <https://www.anthropic.com/engineering/contextual-retrieval> — cited in [Ch 4](../ch04/README.md#references)
- "Controlling Text Search" — PostgreSQL docs. <https://www.postgresql.org/docs/current/textsearch-controls.html> — cited in [Ch 4](../ch04/README.md#references), [Ch 22 › C2](../ch22/README.md#c2-websearch_to_tsquery-ands-every-word-zero-hits)
- "Corrective Retrieval Augmented Generation" — Yan, Gu, Zhu, Ling, arXiv, 2024-01-29. <https://arxiv.org/abs/2401.15884> — cited in [Ch 5](../ch05/README.md#references)
- cross-encoder/ms-marco-MiniLM-L6-v2 model card — Hugging Face. <https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2> — cited in [Ch 4](../ch04/README.md#references)
- Magesh et al., "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools" — Stanford RegLab, 2024 (Journal of Empirical Legal Studies, 2025). <https://reglab.stanford.edu/publications/hallucination-free-assessing-the-reliability-of-leading-ai-legal-research-tools> — cited in [Ch 4](../ch04/README.md#references)
- nomic-embed-text-v1.5 model card (task prefixes) — Nomic AI, Hugging Face. <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5> — cited in [Ch 4](../ch04/README.md#references), [Ch 15](../ch15/README.md#references)
- pgvector README (filtering, iterative index scans) and CHANGELOG (0.8.7, 2026-10-01) — GitHub, accessed 2026-11-07. <https://github.com/pgvector/pgvector> — cited in [Ch 4](../ch04/README.md#references), [Ch 15](../ch15/README.md#references), [Ch 22 › C1](../ch22/README.md#c1-filtered-hnsw-query-returns-fewer-rows-than-limit)
- Cormack, Clarke and Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" — SIGIR 2009. <https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf> — cited in [Ch 4](../ch04/README.md#references)
- Xu et al., "Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering" — LinkedIn, SIGIR 2024. <https://arxiv.org/abs/2404.17723> — cited in [Ch 4](../ch04/README.md#references)
- Sentence Transformers: pretrained cross-encoders — SBERT docs, accessed 2026-11-07. <https://www.sbert.net/docs/cross_encoder/pretrained_models.html> — cited in [Ch 4](../ch04/README.md#references)

## Text-to-SQL & semantic layers

- "A Benchmark to Understand the Role of Knowledge Graphs on Large Language Model's Accuracy for Question Answering on Enterprise SQL Databases" — Sequeda, Allemang, Jacob, arXiv, 2023-11-13. <https://arxiv.org/abs/2311.07509> — cited in [Ch 7](../ch07/README.md#references)
- apache/ossie (formerly Open Semantic Interchange) — GitHub, accessed 2026-11-07. <https://github.com/apache/ossie> — cited in [Ch 7](../ch07/README.md#references)
- Bolt for Python: Socket Mode — Slack, accessed 2026-11-07. <https://docs.slack.dev/tools/bolt-python/concepts/socket-mode> — cited in [Ch 6](../ch06/README.md#references)
- "Build a SQL agent" — LangChain docs, accessed 2026-11-07. <https://docs.langchain.com/oss/python/langchain/sql-agent> — cited in [Ch 6](../ch06/README.md#references)
- Canner/WrenAI — GitHub, accessed 2026-11-07. <https://github.com/canner/wrenai> — cited in [Ch 6](../ch06/README.md#references)
- Cube docs: Core deployment, Postgres data source, environment variables, cubes, REST query format — accessed 2026-11-07. <https://docs.cube.dev/admin/deployment/core.md> · <https://docs.cube.dev/admin/connect-to-data/data-sources/postgres.md> · <https://docs.cube.dev/reference/configuration/environment-variables.md> · <https://docs.cube.dev/docs/data-modeling/cubes.md> · <https://docs.cube.dev/reference/core-data-apis/rest-api/query-format.md> — cited in [Ch 7](../ch07/README.md#references)
- "Cube launches D3, the first agentic analytics platform built on a universal semantic layer" — Cube / PR Newswire, 2025-06-02. <https://www.prnewswire.com/news-releases/cube-launches-d3-the-first-agentic-analytics-platform-built-on-a-universal-semantic-layer-302469860.html> — cited in [Ch 7](../ch07/README.md#references)
- "dbt Labs Affirms Commitment to Open Semantic Interchange by Open Sourcing MetricFlow" — dbt Labs, 2025-10-14. <https://www.getdbt.com/blog/dbt-labs-affirms-commitment-to-open-semantic-interchange-by-open-sourcing-metricflow> — cited in [Ch 7](../ch07/README.md#references)
- dbt-labs/dbt-llm-sl-bench — GitHub, accessed 2026-11-07. <https://github.com/dbt-labs/dbt-llm-sl-bench> — cited in [Ch 7](../ch07/README.md#references)
- "How Airbnb achieved metric consistency at scale" (Part I: Introducing Minerva) — Robert Chang et al., The Airbnb Tech Blog, 2021-04-30. <https://medium.com/airbnb-engineering/how-airbnb-achieved-metric-consistency-at-scale-f23cc53dea70> — cited in [Ch 7](../ch07/README.md#references)
- "How we built Text-to-SQL at Pinterest" — Pinterest Engineering, 2024-04-02. <https://medium.com/pinterest-engineering/how-we-built-text-to-sql-at-pinterest-30bad30dabff> — cited in [Ch 6](../ch06/README.md#references)
- "Metric views" — Databricks docs, updated 2026-10-29. <https://docs.databricks.com/aws/en/uc-semantics/metric-views/> — cited in [Ch 7](../ch07/README.md#references)
- "MetricFlow time spine" — dbt docs, accessed 2026-11-07. <https://docs.getdbt.com/docs/build/metricflow-time-spine> — cited in [Ch 7](../ch07/README.md#references)
- "Metrics overview" — dbt docs, accessed 2026-11-07. <https://docs.getdbt.com/docs/build/metrics-overview> — cited in [Ch 7](../ch07/README.md#references)
- PostgreSQL docs: GRANT (column privileges), `default_transaction_read_only`, `statement_timeout`. <https://www.postgresql.org/docs/current/sql-grant.html> — cited in [Ch 6](../ch06/README.md#references)
- "Practical text-to-SQL for data analytics" — LinkedIn Engineering, 2024. <https://www.linkedin.com/blog/engineering/ai/practical-text-to-sql-for-data-analytics> — cited in [Ch 6](../ch06/README.md#references)
- "QueryGPT – Natural Language to SQL Using Generative AI" — Uber Engineering, 2024-09-19. <https://www.uber.com/en-DE/blog/query-gpt/> — cited in [Ch 6](../ch06/README.md#references)
- "Semantic Layer as the Data Interface for LLMs" — dbt Labs, 2023-11-30. <https://www.getdbt.com/blog/semantic-layer-as-the-data-interface-for-llms> — cited in [Ch 7](../ch07/README.md#references)
- "Semantic Layer vs. Text-to-SQL: 2026 Benchmark Update" — Jason Ganz, Benoit Perigaud, dbt Developer Blog, 2026-04-07. <https://docs.getdbt.com/blog/semantic-layer-vs-text-to-sql-2026> — cited in [Ch 7](../ch07/README.md#references)
- "Semantic Layers for Reliable LLM-Powered Data Analytics: A Paired Benchmark of Accuracy and Hallucination Across Three Frontier Models" — Rumiantsau, Fokeev, arXiv, 2026-04-28. <https://arxiv.org/abs/2604.25149> — cited in [Ch 7](../ch07/README.md#references)
- "Semantic views overview" — Snowflake docs, accessed 2026-11-07. <https://docs.snowflake.com/en/user-guide/views-semantic/overview> — cited in [Ch 7](../ch07/README.md#references)
- "Snowflake Cortex Analyst: Behind the Scenes" — Snowflake Engineering, 2024-08-14. <https://www.snowflake.com/en/engineering-blog/snowflake-cortex-analyst-behind-the-scenes/> — cited in [Ch 7](../ch07/README.md#references)
- "Snowflake, Salesforce, dbt Labs, and More Revolutionize Data Readiness for AI with Open Semantic Interchange Initiative" — Snowflake, 2025-09-23. <https://www.snowflake.com/en/news/press-releases/snowflake-salesforce-dbt-labs-and-more-revolutionize-data-readiness-for-ai-with-open-semantic-interchange-initiative/> — cited in [Ch 7](../ch07/README.md#references)
- sqlglot — GitHub. <https://github.com/tobymao/sqlglot> — cited in [Ch 5](../ch05/README.md#references), [Ch 6](../ch06/README.md#references)
- sqlglot releases and expression builders — GitHub, accessed 2026-11-07. <https://github.com/tobymao/sqlglot/releases> · <https://sqlglot.com/sqlglot/expressions.html> — cited in [Ch 7](../ch07/README.md#references)
- vanna-ai/vanna (archived) — GitHub, accessed 2026-11-07. <https://github.com/vanna-ai/vanna> — cited in [Ch 6](../ch06/README.md#references)

## Documents & multimodal input

- "Advancing Invoice Document Processing at Uber using GenAI" — Uber Engineering, 2025-04-17. <https://www.uber.com/in/en/blog/advancing-invoice-document-processing-using-genai/> — cited in [Ch 8](../ch08/README.md#references)
- "Financial benchmarks" — Ramp Builders, 2026-03-09. <https://builders.ramp.com/post/financial-benchmarks> — cited in [Ch 8](../ch08/README.md#references)
- Gemma 3 model card — Google AI for Developers, accessed 2026-11-07. <https://ai.google.dev/gemma/docs/core/model_card_3> — cited in [Ch 8](../ch08/README.md#references)
- glm-ocr — Ollama library, accessed 2026-11-07. <https://ollama.com/library/glm-ocr> — cited in [Ch 8](../ch08/README.md#references)
- "GLM-OCR technical report" — Zhipu AI & Tsinghua University, arXiv 2603.10910, 2026-03-11. <https://arxiv.org/abs/2603.10910> — cited in [Ch 8](../ch08/README.md#references)
- Images and vision — OpenAI docs, accessed 2026-11-07. <https://developers.openai.com/api/docs/guides/images-vision> — cited in [Ch 8](../ch08/README.md#references)
- "Lithuanian Man Sentenced To 5 Years In Prison For Theft Of Over $120 Million In Fraudulent Business Email Compromise Scheme" — U.S. Attorney's Office SDNY, 2019-12-19 (archived). <https://web.archive.org/web/20210117074119/https://www.justice.gov/usao-sdny/pr/lithuanian-man-sentenced-5-years-prison-theft-over-120-million-fraudulent-business> — cited in [Ch 8](../ch08/README.md#references)
- Ollama docs: Vision, OpenAI compatibility, Context length — accessed 2026-11-07. <https://docs.ollama.com/capabilities/vision> · <https://docs.ollama.com/api/openai-compatibility> · <https://docs.ollama.com/context-length> — cited in [Ch 8](../ch08/README.md#references), [Ch 16](../ch16/README.md#references)
- OmniDocBench (paper and leaderboard) — OpenDataLab, arXiv 2412.07626; README accessed 2026-11-07. <https://github.com/opendatalab/OmniDocBench> — cited in [Ch 8](../ch08/README.md#references)
- "Over 75% of U.S. Firms Experienced Payments Fraud in 2025…" (AFP 2026 Payments Fraud and Control Survey) — PR Newswire, 2026-04-14. <https://www.prnewswire.com/news-releases/over-75-of-us-firms-experienced-payments-fraud-in-2025-while-ai-adoption-for-fraud-mitigation-lags-302738857.html> — cited in [Ch 8](../ch08/README.md#references)
- pypdf: Extract text from a PDF — pypdf docs, accessed 2026-11-07. <https://pypdf.readthedocs.io/en/stable/user/extract-text.html> — cited in [Ch 8](../ch08/README.md#references)
- "Xerox scanners/photocopiers randomly alter numbers in scanned documents" — D. Kriesel, 2013-08-02. <https://www.dkriesel.com/en/blog/2013/0802_xerox-workcentres_are_switching_written_numbers_when_scanning> — cited in [Ch 8](../ch08/README.md#references)
- zai-org/GLM-OCR model card — Hugging Face, accessed 2026-11-07. <https://huggingface.co/zai-org/GLM-OCR> — cited in [Ch 8](../ch08/README.md#references)

## Agents, tools & MCP

- "Build a custom RAG agent with LangGraph" — LangChain docs, accessed 2026-11-07. <https://docs.langchain.com/oss/python/langgraph/agentic-rag> — cited in [Ch 5](../ch05/README.md#references)
- "Building effective agents" — Anthropic, 2024-12-19. <https://www.anthropic.com/research/building-effective-agents> — cited in [Ch 5](../ch05/README.md#references), [Ch 10](../ch10/README.md#references)
- "Checkpointers" — LangChain docs, accessed 2026-11-07. <https://docs.langchain.com/oss/python/langgraph/checkpointers> — cited in [Ch 5](../ch05/README.md#references)
- "Code execution with MCP: building more efficient AI agents" — Anthropic Engineering, 2025-11-04. <https://www.anthropic.com/engineering/code-execution-with-mcp> — cited in [Ch 9](../ch09/README.md#references), [Ch 22 › I2](../ch22/README.md#i2-tool-schemas-tax-every-call)
- "Don't Build Multi-Agents" — Walden Yan, Cognition, 2025-06-12. <https://cognition.ai/blog/dont-build-multi-agents> — cited in [Ch 10](../ch10/README.md#references)
- "Effective context engineering for AI agents" — Anthropic Engineering, 2025-09-29. <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents> — cited in [Ch 10](../ch10/README.md#references)
- "Function calling", section Token usage — OpenAI docs. <https://developers.openai.com/api/docs/guides/function-calling> — cited in [Ch 22 › I2](../ch22/README.md#i2-tool-schemas-tax-every-call)
- "Graph API overview" — LangChain docs, accessed 2026-11-07. <https://docs.langchain.com/oss/python/langgraph/graph-api> — cited in [Ch 5](../ch05/README.md#references)
- "GRAPH_RECURSION_LIMIT" — LangChain docs, accessed 2026-11-07. <https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT> — cited in [Ch 5](../ch05/README.md#references)
- "How Klarna's AI assistant redefined customer support at scale for 85 million active users" — LangChain, 2025-02-12. <https://www.langchain.com/blog/customers-klarna> — cited in [Ch 5](../ch05/README.md#references)
- "How Lyft Built a Self-Serve AI Agent Platform for Customer Support with LangGraph and LangSmith" — Akshay Sharma (Lyft), LangChain blog, 2026-05-27. <https://www.langchain.com/blog/lyft-built-a-self-serve-ai-agent-platform-for-customer-support-with-langgraph-and-langsmith> — cited in [Ch 5](../ch05/README.md#references)
- "How we built our multi-agent research system" — Anthropic Engineering, 2025-06-13. <https://www.anthropic.com/engineering/multi-agent-research-system> — cited in [Ch 10](../ch10/README.md#references)
- Human-in-the-loop — OpenAI Agents SDK docs, accessed 2026-11-07. <https://openai.github.io/openai-agents-python/human_in_the_loop/> — cited in [Ch 10](../ch10/README.md#references)
- "Idempotent requests" — Stripe API reference, accessed 2026-11-07. <https://docs.stripe.com/api/idempotent_requests> — cited in [Ch 9](../ch09/README.md#references), [Ch 22 › D2](../ch22/README.md#d2-retry--duplicate-side-effects)
- "Interrupts" — LangGraph docs, accessed 2026-11-07. <https://docs.langchain.com/oss/python/langgraph/interrupts> — cited in [Ch 10](../ch10/README.md#references), [Ch 22 › D4](../ch22/README.md#d4-langgraph-interrupt-re-runs-the-node-on-resume)
- langchain-ai/langgraph — GitHub. <https://github.com/langchain-ai/langgraph> — cited in [Ch 5](../ch05/README.md#references)
- langgraph-checkpoint-postgres README — GitHub, accessed 2026-11-07. <https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres> — cited in [Ch 10](../ch10/README.md#references)
- MCP specification 2026-07-28: stdio transport — modelcontextprotocol.io, accessed 2026-11-07. <https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio> — cited in [Ch 9](../ch09/README.md#references)
- MCP specification 2026-07-28: Tools — modelcontextprotocol.io, accessed 2026-11-07. <https://modelcontextprotocol.io/specification/2026-07-28/server/tools> — cited in [Ch 9](../ch09/README.md#references)
- "Migration Guide: v1 to v2" — MCP Python SDK docs, accessed 2026-11-07. <https://py.sdk.modelcontextprotocol.io/migration/> — cited in [Ch 9](../ch09/README.md#references)
- Models (third-party providers, tracing 401) — OpenAI Agents SDK docs, accessed 2026-11-07. <https://openai.github.io/openai-agents-python/models/> — cited in [Ch 10](../ch10/README.md#references)
- Persistence — LangGraph docs, accessed 2026-11-07. <https://docs.langchain.com/oss/python/langgraph/persistence> — cited in [Ch 10](../ch10/README.md#references)
- ShopFlow "Refund Approval Limits and Process" — this repo. [datasets/shopflow/docs/handbook/refund-approval-limits.md](../datasets/shopflow/docs/handbook/refund-approval-limits.md) — cited in [Ch 9](../ch09/README.md#references), [Ch 10](../ch10/README.md#references)
- "Tool use with Claude" — Anthropic docs. <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview> — cited in [Ch 22 › I2](../ch22/README.md#i2-tool-schemas-tax-every-call)
- "What's new in v2" — MCP Python SDK docs, accessed 2026-11-07. <https://py.sdk.modelcontextprotocol.io/whats-new/> — cited in [Ch 9](../ch09/README.md#references)

## Security & guardrails

- "Breaking down 'EchoLeak'" — Aim Labs, 2025-06-11 (archived). <http://web.archive.org/web/20250805110050/https://www.aim.security/lp/aim-labs-echoleak-blogpost> — cited in [Ch 13](../ch13/README.md#references)
- "CaMeL offers a promising new direction for mitigating prompt injection attacks" — Simon Willison, 2025-04-11. <https://simonwillison.net/2025/Apr/11/camel/> — cited in [Ch 13](../ch13/README.md#references)
- "CoSnitch: When Your AI Assistant Becomes Its Own Whistleblower" — Varonis, 2026. <https://www.varonis.com/blog/cosnitch> — cited in [Ch 13](../ch13/README.md#references)
- CVE-2025-32711 — NVD. <https://nvd.nist.gov/vuln/detail/CVE-2025-32711> — cited in [Ch 13](../ch13/README.md#references)
- "Data Exfiltration from Slack AI via Indirect Prompt Injection" — PromptArmor, 2024-08-20. <https://www.promptarmor.com/resources/data-exfiltration-from-slack-ai-via-indirect-prompt-injection> — cited in [Ch 13](../ch13/README.md#references), [Ch 22 › H2](../ch22/README.md#h2-prompt-injection-via-retrieved-content)
- "Defeating Prompt Injections by Design" (CaMeL) — Debenedetti et al., arXiv 2503.18813, 2025-03-24. <https://arxiv.org/abs/2503.18813> — cited in [Ch 13](../ch13/README.md#references)
- "Design Patterns for Securing LLM Agents against Prompt Injections" — Beurer-Kellner et al., arXiv 2506.08837, 2025-06-10. <https://arxiv.org/abs/2506.08837> — cited in [Ch 13](../ch13/README.md#references)
- "GitHub MCP Exploited: Accessing private repositories via MCP" — Invariant Labs, 2025-05-26. <https://invariantlabs.ai/blog/mcp-github-vulnerability> — cited in [Ch 9](../ch09/README.md#references), [Ch 13](../ch13/README.md#references), [Ch 22 › H2](../ch22/README.md#h2-prompt-injection-via-retrieved-content)
- "Hacking Google Bard - From Prompt Injection to Data Exfiltration" — Johann Rehberger, Embrace The Red, 2023-11-03. <https://embracethered.com/blog/posts/2023/google-bard-data-exfiltration/> — cited in [Ch 13](../ch13/README.md#references)
- "How We Exploited CodeRabbit: From a Simple PR to RCE and Write Access on 1M Repositories" — Kudelski Security, 2025-08-19. <https://kudelskisecurity.com/research/how-we-exploited-coderabbit-from-a-simple-pr-to-rce-and-write-access-on-1m-repositories> — cited in [Ch 12](../ch12/README.md#references)
- "Keeping your GitHub Actions and workflows secure Part 1: Preventing pwn requests" — GitHub Security Lab, 2021-08-03. <https://securitylab.github.com/resources/github-actions-preventing-pwn-requests/> — cited in [Ch 12](../ch12/README.md#references)
- "LiteLLM compromised on PyPI: TeamPCP supply chain campaign" — Datadog Security Labs, 2026-03-24. <https://securitylabs.datadoghq.com/articles/litellm-compromised-pypi-teampcp-supply-chain-campaign/> — cited in [Ch 15](../ch15/README.md#references)
- Llama Guard 3 1B model card — Meta PurpleLlama. <https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/1B/MODEL_CARD.md> — cited in [Ch 13](../ch13/README.md#references)
- Llama Guard 3 prompt format — Meta, accessed 2026-11-07. <https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/> — cited in [Ch 13](../ch13/README.md#references)
- Llama Prompt Guard 2 model card — Meta PurpleLlama. <https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Prompt-Guard-2/86M/MODEL_CARD.md> — cited in [Ch 13](../ch13/README.md#references)
- llama-guard3 — Ollama library, accessed 2026-11-07. <https://ollama.com/library/llama-guard3> — cited in [Ch 13](../ch13/README.md#references)
- "LLM01:2025 Prompt Injection" — OWASP GenAI. <https://genai.owasp.org/llmrisk/llm01-prompt-injection/> — cited in [Ch 22 › H2](../ch22/README.md#h2-prompt-injection-via-retrieved-content)
- "LLM02:2025 Sensitive Information Disclosure" — OWASP GenAI. <https://genai.owasp.org/llmrisk/llm022025-sensitive-information-disclosure/> — cited in [Ch 22 › H1](../ch22/README.md#h1-pii-in-logs)
- Microsoft Presidio (PII detection and anonymization) — GitHub. <https://github.com/microsoft/presidio> — cited in [Ch 14](../ch14/README.md#references)
- nemoguardrails 0.24.1, guardrails-ai 0.11.0, llamafirewall 1.0.3, llm-guard 0.3.16, presidio-analyzer 2.2.366 — PyPI, accessed 2026-11-07. <https://pypi.org/project/nemoguardrails/> · <https://pypi.org/project/guardrails-ai/> · <https://pypi.org/project/llamafirewall/> · <https://pypi.org/project/llm-guard/> · <https://pypi.org/project/presidio-analyzer/> — cited in [Ch 13](../ch13/README.md#references)
- "OWASP GenAI LLM Top 10 2026" — OWASP GenAI Security Project, 2026-08-03. <https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/> — cited in [Ch 13](../ch13/README.md#references)
- "OWASP Top 10 for Agentic Applications for 2026" — OWASP GenAI Security Project, 2025-12-09. <https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/> — cited in [Ch 13](../ch13/README.md#references)
- "OWASP Top 10 for LLM Applications 2025" — OWASP GenAI Security Project, 2024-11-17. <https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/> — cited in [Ch 13](../ch13/README.md#references)
- RFC 2606: Reserved Top Level DNS Names — IETF, 1999. <https://www.rfc-editor.org/rfc/rfc2606> — cited in [Ch 13](../ch13/README.md#references)
- Securely using pull_request_target — GitHub Docs, accessed 2026-11-07. <https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target> — cited in [Ch 12](../ch12/README.md#references)
- "Supabase MCP can leak your entire SQL database" — General Analysis, 2025-07-08 (reviewed 2026-10-21). <https://generalanalysis.com/blog/supabase-mcp-blog> — cited in [Ch 9](../ch09/README.md#references), [Ch 13](../ch13/README.md#references)
- "The Dual LLM pattern for building AI assistants that can resist prompt injection" — Simon Willison, 2023-04-25. <https://simonwillison.net/2023/Apr/25/dual-llm-pattern/> — cited in [Ch 13](../ch13/README.md#references)
- "The lethal trifecta for AI agents: private data, untrusted content, and external communication" — Simon Willison, 2025-06-16. <https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/> — cited in [Ch 13](../ch13/README.md#references), [Ch 22 › H2](../ch22/README.md#h2-prompt-injection-via-retrieved-content)

## Observability & incident analysis

- Arize Phoenix: Docker deployment — Arize docs, accessed 2026-11-07. <https://arize.com/docs/phoenix/self-hosting/deployment-options/docker> — cited in [Ch 14](../ch14/README.md#references)
- Chen et al., "Automatic Root Cause Analysis via Large Language Models for Cloud Incidents" (RCACopilot) — EuroSys 2024, arXiv 2305.15778. <https://arxiv.org/abs/2305.15778> — cited in [Ch 11](../ch11/README.md#references)
- "ClickHouse raises $400M Series D ... acquires Langfuse" — ClickHouse, 2026-01-16. <https://clickhouse.com/blog/clickhouse-raises-400-million-series-d-acquires-langfuse-launches-postgres> — cited in [Ch 14](../ch14/README.md#references)
- "Dealing with Missing Time Series in Prometheus" — PromLabs, 2023-09-13. <https://promlabs.com/blog/2023/09/13/dealing-with-missing-time-series-in-prometheus/> — cited in [Ch 11](../ch11/README.md#references), [Ch 22 › J2](../ch22/README.md#j2-labeled-prometheus-counter-missing-until-first-increment)
- Grafana Alloy documentation — Grafana Labs, accessed 2026-11-07. <https://grafana.com/docs/alloy/latest/> — cited in [Ch 11](../ch11/README.md#references)
- "Instrumentation: avoid missing metrics" — Prometheus docs. <https://prometheus.io/docs/practices/instrumentation/> — cited in [Ch 22 › J2](../ch22/README.md#j2-labeled-prometheus-counter-missing-until-first-increment)
- "Introduction" (Change Management: "roughly 70% of outages are due to changes in a live system") — Google, *Site Reliability Engineering*, 2016. <https://sre.google/sre-book/introduction/> — cited in [Ch 11](../ch11/README.md#references)
- Label best practices — Grafana Loki docs, accessed 2026-11-07. <https://grafana.com/docs/loki/latest/get-started/labels/bp-labels/> — cited in [Ch 11](../ch11/README.md#references)
- "Labels" — Prometheus client_python docs. <https://prometheus.github.io/client_python/instrumenting/labels/> — cited in [Ch 22 › J2](../ch22/README.md#j2-labeled-prometheus-counter-missing-until-first-increment)
- Langfuse architecture (queued ingestion, S3 first) — Langfuse docs, accessed 2026-11-07. <https://langfuse.com/self-hosting> — cited in [Ch 14](../ch14/README.md#references)
- Langfuse self-hosting: Docker Compose deployment — Langfuse docs, accessed 2026-11-07. <https://langfuse.com/self-hosting/deployment/docker-compose> — cited in [Ch 14](../ch14/README.md#references)
- langfuse/langfuse `docker-compose.yml` at v4.76.0 — GitHub. <https://github.com/langfuse/langfuse/blob/v4.76.0/docker-compose.yml> — cited in [Ch 14](../ch14/README.md#references)
- "Leveraging AI for efficient incident response" — Engineering at Meta, 2024-06-24. <https://engineering.fb.com/2024/06/24/data-infrastructure/leveraging-ai-for-efficient-incident-response/> — cited in [Ch 11](../ch11/README.md#references)
- Loki HTTP API reference (`POST /loki/api/v1/push`, `GET /loki/api/v1/query_range`) — Grafana Labs, accessed 2026-11-07. <https://grafana.com/docs/loki/latest/reference/loki-http-api/> — cited in [Ch 11](../ch11/README.md#references)
- OpenTelemetry GenAI attribute registry (`gen_ai.usage.input_tokens` SHOULD include cached tokens) — GitHub, accessed 2026-11-08. <https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md> — cited in [Ch 14](../ch14/README.md#references)
- OpenTelemetry GenAI semantic conventions: inference span reference — GitHub, accessed 2026-11-07. <https://github.com/open-telemetry/semantic-conventions-genai/blob/main/reference/reports/inference-span.md> — cited in [Ch 14](../ch14/README.md#references)
- OpenTelemetry semantic conventions releases (v1.37.0 message attributes; v1.42.0 GenAI move) — GitHub. <https://github.com/open-telemetry/semantic-conventions/releases> — cited in [Ch 14](../ch14/README.md#references)
- Prometheus Python client — prometheus.github.io, accessed 2026-11-07. <https://prometheus.github.io/client_python/> — cited in [Ch 11](../ch11/README.md#references)
- Promtail agent (end of life as of 2026-03-02) — Grafana Loki docs, accessed 2026-11-07. <https://grafana.com/docs/loki/latest/send-data/promtail/> — cited in [Ch 11](../ch11/README.md#references)
- "Python SDK v5 and JS/TS SDK v6" — Langfuse changelog, 2026-10-14. <https://langfuse.com/changelog/2026-10-14-python-sdk-v5-js-sdk-v6> — cited in [Ch 14](../ch14/README.md#references)
- Python v4 → v5 migration guide — Langfuse docs, accessed 2026-11-07. <https://langfuse.com/docs/observability/sdk/upgrade-path/python-v4-to-v5> — cited in [Ch 14](../ch14/README.md#references)
- User feedback — Langfuse docs, accessed 2026-11-07. <https://langfuse.com/docs/observability/features/user-feedback> — cited in [Ch 14](../ch14/README.md#references)
- Versions & Compatibility (feature availability matrix) — Langfuse docs, accessed 2026-11-07. <https://langfuse.com/self-hosting/upgrade/versioning> — cited in [Ch 14](../ch14/README.md#references)

## Gateways, caching & cost

- "Cutting our inference bill by 68% by being obsessive about optimization" — Cloudflare blog, 2026-10-19. <https://blog.cloudflare.com/obsessive-inference-optimization> — cited in [Ch 12](../ch12/README.md#references)
- Chen, Zaharia, Zou. "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance" — arXiv, 2023-05-09. <https://arxiv.org/abs/2305.05176> — cited in [Ch 15](../ch15/README.md#references)
- "How a timestamp was busting our prompt cache" — Basedash, 2026-07-07. <https://www.basedash.com/blog/how-a-timestamp-was-busting-our-prompt-cache> — cited in [Ch 17](../ch17/README.md#references)
- LiteLLM docs: proxy config, caching, semantic caching, virtual keys, reliability/fallbacks, deploy — accessed 2026-11-07. <https://docs.litellm.ai/docs/proxy/configs> · <https://docs.litellm.ai/docs/proxy/caching> · <https://docs.litellm.ai/docs/proxy/caching_semantic> · <https://docs.litellm.ai/docs/proxy/virtual_keys> · <https://docs.litellm.ai/docs/proxy/reliability> · <https://docs.litellm.ai/docs/proxy/deploy> — cited in [Ch 15](../ch15/README.md#references)
- LiteLLM releases (v1.108.2 LTS) — GitHub, accessed 2026-11-07. <https://github.com/BerriAI/litellm/releases> — cited in [Ch 15](../ch15/README.md#references)
- "LLM inference prices have fallen rapidly but unequally across tasks" — Epoch AI, 2025-03. <https://epoch.ai/data-insights/llm-inference-price-trends> — cited in [Ch 22](../ch22/README.md#references)
- "Prompt caching" — Anthropic docs, accessed 2026-11-07. <https://platform.claude.com/docs/en/build-with-claude/prompt-caching> (also cited as <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>, which redirects here) — cited in [Ch 15](../ch15/README.md#references), [Ch 17](../ch17/README.md#references), [Ch 22 › I1](../ch22/README.md#i1-prompt-cache-invalidation)
- "Prompt caching" — OpenAI docs, accessed 2026-11-07. <https://developers.openai.com/api/docs/guides/prompt-caching> — cited in [Ch 15](../ch15/README.md#references), [Ch 22 › I1](../ch22/README.md#i1-prompt-cache-invalidation)
- "Prompt caching" — OpenRouter docs, accessed 2026-11-07. <https://openrouter.ai/docs/features/prompt-caching> — cited in [Ch 15](../ch15/README.md#references)
- "Prompt caching with Claude" — Anthropic, 2024. <https://www.anthropic.com/news/prompt-caching> — cited in [Ch 15](../ch15/README.md#references)
- "RouteLLM: An Open-Source Framework for Cost-Effective LLM Routing" — LMSYS, 2024-07-01. <https://lmsys.org/blog/2024-07-01-routellm/> — cited in [Ch 15](../ch15/README.md#references)
- "Semantic caching for LLM apps" — Vercel, 2026-08-03. <https://vercel.com/i/semantic-caching-llm-apps> — cited in [Ch 15](../ch15/README.md#references)
- "The plunging price of thought" — Epoch AI, 2026. <https://epoch.ai/publications/the-plunging-price-of-thought> — cited in [Ch 22](../ch22/README.md#references)
- "Welcome to LLMflation – LLM inference cost is going down fast" — a16z, 2024-11-12. <https://a16z.com/llmflation-llm-inference-cost/> — cited in [Ch 22](../ch22/README.md#references)

## Serving & GPUs

- "Automatic Prefix Caching" — vLLM docs, accessed 2026-11-07. <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/> — cited in [Ch 17](../ch17/README.md#references)
- Best practices for autoscaling LLM inference workloads with GPUs — Google Cloud (GKE docs). <https://cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling> — cited in [Ch 20](../ch20/README.md#references)
- "CVE-2026-22778: critical RCE vulnerability in vLLM" — Orca Security, 2026-02-02. <https://orca.security/resources/blog/cve-2026-22778-vllm-rce-vulnerability/> — cited in [Ch 16](../ch16/README.md#references)
- "Defeating Nondeterminism in LLM Inference" — Thinking Machines (Horace He), 2025-09-10. <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/> — cited in [Ch 22 › B2](../ch22/README.md#b2-nondeterminism-at-temperature-0)
- "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving" — arXiv, 2024-01-18. <https://arxiv.org/abs/2401.09670> — cited in [Ch 17](../ch17/README.md#references)
- "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test" — arXiv, 2025-03-03. <https://arxiv.org/abs/2503.01840> — cited in [Ch 17](../ch17/README.md#references)
- "Efficient Memory Management for Large Language Model Serving with PagedAttention" — Kwon et al., SOSP 2023. <https://arxiv.org/abs/2309.06180> — cited in [Ch 17](../ch17/README.md#references)
- "Engine Arguments" — vLLM docs, accessed 2026-11-07. <https://docs.vllm.ai/en/latest/configuration/engine_args/> — cited in [Ch 16](../ch16/README.md#references), [Ch 17](../ch17/README.md#references)
- "Fast and expressive LLM inference with RadixAttention and SGLang" — LMSYS, 2024-01-17. <https://lmsys.org/blog/2024-01-17-sglang/> — cited in [Ch 16](../ch16/README.md#references)
- "Fast Inference from Transformers via Speculative Decoding" — Leviathan et al., ICML 2023. <https://arxiv.org/abs/2211.17192> — cited in [Ch 17](../ch17/README.md#references)
- "FP8 W8A8 / W8A16" — vLLM docs, accessed 2026-11-07. <https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/> — cited in [Ch 17](../ch17/README.md#references)
- Gateway API Inference Extension: InferencePool — Kubernetes SIGs, accessed 2026-11-07. <https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferencepool/> — cited in [Ch 16](../ch16/README.md#references)
- "GPUStack performance lab: gpt-oss-120b on H100" — GPUStack, accessed 2026-11-07. <https://github.com/gpustack/gpustack/blob/main/docs/performance-lab/gpt-oss-120b/h100.md> — cited in [Ch 17](../ch17/README.md#references)
- "GPUStack performance lab: Qwen3-8B on H100 (latency)" — GPUStack, accessed 2026-11-07. <https://github.com/gpustack/gpustack/blob/main/docs/performance-lab/qwen3-8b/h100-latency.md> — cited in [Ch 17](../ch17/README.md#references)
- "How Amazon scaled Rufus by building multi-node inference using AWS Trainium chips and vLLM" — AWS Machine Learning Blog, 2025-08-13. <https://aws.amazon.com/blogs/machine-learning/how-amazon-scaled-rufus-by-building-multi-node-inference-using-aws-trainium-chips-and-vllm/> — cited in [Ch 16](../ch16/README.md#references)
- "How continuous batching enables 23x throughput in LLM inference" — Anyscale, 2023-06-22. <https://www.anyscale.com/blog/continuous-batching-llm-inference> — cited in [Ch 17](../ch17/README.md#references)
- "How llm-d makes the most of the hardware you already have" — IBM Research, 2026-09-08. <https://research.ibm.com/blog/running-open-models-on-h100-gpus-with-llmd> — cited in [Ch 16](../ch16/README.md#references)
- "How we leveraged vLLM to power our GenAI applications" — LinkedIn Engineering, 2025-08-26. <https://www.linkedin.com/blog/engineering/ai/how-we-leveraged-vllm-to-power-our-genai-applications> — cited in [Ch 16](../ch16/README.md#references)
- "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference" — NVIDIA Technical Blog, 2025-06-24. <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/> — cited in [Ch 17](../ch17/README.md#references)
- "llama.cpp HTTP Server" README — ggml-org, accessed 2026-11-07. <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md> — cited in [Ch 16](../ch16/README.md#references), [Ch 17](../ch17/README.md#references)
- llm-d Quickstart (v1.0) — llm-d, accessed 2026-11-07. <https://llm-d.ai/docs/getting-started/quickstart> — cited in [Ch 16](../ch16/README.md#references)
- llm-d repository — GitHub, accessed 2026-11-07. <https://github.com/llm-d/llm-d> — cited in [Ch 16](../ch16/README.md#references)
- "llm-d v1.0: state-of-the-art inference, ready for day 2" — llm-d blog, 2026-10-20. <https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2> — cited in [Ch 16](../ch16/README.md#references)
- "Looking back at speculative decoding" — Google Research, 2024-12-06. <https://research.google/blog/looking-back-at-speculative-decoding/> — cited in [Ch 17](../ch17/README.md#references)
- "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving" — arXiv, 2024-06-24. <https://arxiv.org/abs/2407.00079> — cited in [Ch 17](../ch17/README.md#references)
- "NVIDIA Ampere GA102 GPU Architecture" whitepaper — NVIDIA. <https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf> — cited in [Ch 17](../ch17/README.md#references)
- NVIDIA/nvkind — NVIDIA. <https://github.com/NVIDIA/nvkind> — cited in [Ch 20](../ch20/README.md#references)
- Ollama FAQ (`OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_QUEUE`) — Ollama docs, accessed 2026-11-07. <https://docs.ollama.com/faq> — cited in [Ch 16](../ch16/README.md#references)
- "Ollama vs. vLLM: a deep dive into performance benchmarking" — Red Hat Developer, 2025-08-08. <https://developers.redhat.com/articles/2025/08/08/ollama-vs-vllm-deep-dive-performance-benchmarking> — cited in [Ch 16](../ch16/README.md#references)
- "openai/gpt-oss-120b" model card — OpenAI on Hugging Face, 2025. <https://huggingface.co/openai/gpt-oss-120b> — cited in [Ch 17](../ch17/README.md#references)
- "Optimizing AI Inference at Character.AI" — Character.AI, 2024-06-20. <https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/> — cited in [Ch 17](../ch17/README.md#references)
- "Orca: A Distributed Serving System for Transformer-Based Generative Models" — USENIX OSDI 2022. <https://www.usenix.org/conference/osdi22/presentation/yu> — cited in [Ch 17](../ch17/README.md#references)
- "Parallelism and Scaling" — vLLM docs, accessed 2026-11-07. <https://docs.vllm.ai/en/latest/serving/parallelism_scaling/> — cited in [Ch 17](../ch17/README.md#references)
- "Quantization" support matrix — vLLM docs, accessed 2026-11-07. <https://docs.vllm.ai/en/latest/features/quantization/> — cited in [Ch 17](../ch17/README.md#references)
- "Red Hat launches the llm-d community" — Red Hat press release, 2025-05-20. <https://www.redhat.com/en/about/press-releases/red-hat-launches-llm-d-community-powering-distributed-gen-ai-inference-scale> — cited in [Ch 16](../ch16/README.md#references)
- "RunPod GPU pricing" (RTX 3090: $0.22/h community, $0.50/h secure) — RunPod, page updated 2026-11-04. <https://www.runpod.io/pricing> — cited in [Ch 16](../ch16/README.md#references), [Ch 17](../ch17/README.md#references)
- SGLang install — SGLang docs, accessed 2026-11-07. <https://docs.sglang.io/docs/get-started/install> — cited in [Ch 16](../ch16/README.md#references)
- "SGLang joins PyTorch ecosystem" — PyTorch blog, 2025. <https://pytorch.org/blog/sglang-joins-pytorch/> — cited in [Ch 16](../ch16/README.md#references)
- SGLang server arguments — SGLang docs, accessed 2026-11-07. <https://docs.sglang.io/docs/advanced_features/server_arguments.md> — cited in [Ch 16](../ch16/README.md#references)
- "Speculative Decoding" — vLLM docs, accessed 2026-11-07. <https://docs.vllm.ai/en/latest/features/speculative_decoding/> — cited in [Ch 17](../ch17/README.md#references)
- Speculative-decoding experiments on an RTX 3090 with llama.cpp and vLLM — HackMD (thc1006), 2026-04, updated 2026-05-17. <https://hackmd.io/ODXuOQNzSiyUITz7g9mtBw> — cited in [Ch 17](../ch17/README.md#references)
- Understanding LLMInferenceService — KServe docs (v0.20), accessed 2026-11-07. <https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview> — cited in [Ch 20](../ch20/README.md#references)
- vLLM benchmark CLI — vLLM docs, accessed 2026-11-07. <https://docs.vllm.ai/en/latest/benchmarking/cli/> — cited in [Ch 16](../ch16/README.md#references)
- vLLM metrics — vLLM docs, accessed 2026-11-07. <https://docs.vllm.ai/en/latest/usage/metrics/> — cited in [Ch 16](../ch16/README.md#references)
- vLLM OpenAI-compatible server — vLLM docs, accessed 2026-11-07. <https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/> — cited in [Ch 16](../ch16/README.md#references)
- vLLM production-stack (Helm chart `vllm-stack` 0.1.14) — vLLM project. <https://github.com/vllm-project/production-stack> — cited in [Ch 20](../ch20/README.md#references)
- vLLM reasoning outputs — vLLM docs, accessed 2026-11-07. <https://docs.vllm.ai/en/latest/features/reasoning_outputs/> — cited in [Ch 16](../ch16/README.md#references)
- "vLLM V1: A Major Upgrade to vLLM's Core Architecture" — vLLM blog, 2025-01-27. <https://blog.vllm.ai/2025/01/27/v1-alpha-release.html> — cited in [Ch 17](../ch17/README.md#references)
- "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention" — vLLM team, 2023-06-20. <https://vllm.ai/blog/2023-06-20-vllm> — cited in [Ch 17](../ch17/README.md#references)

## Fine-tuning

- "3x Faster LLM Training with Unsloth Kernels + Packing" — Unsloth docs, accessed 2026-11-09. <https://unsloth.ai/docs/blog/3x-faster-training-packing> — cited in [Ch 18](../ch18/README.md#references)
- "About supervised fine-tuning for Gemini models" (limits) — Google Cloud, accessed 2026-11-08. <https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/supervised-tuning> — cited in [Ch 19](../ch19/README.md#references)
- "Agent Platform Pricing" — Google Cloud, accessed 2026-11-08. <https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing> — cited in [Ch 19](../ch19/README.md#references)
- "Agent Platform SDK for Python: version 2.0.1 migration guide" — Google Cloud, accessed 2026-11-08. <https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/python-sdk/sdk-migration> — cited in [Ch 19](../ch19/README.md#references)
- "Amazon Bedrock endpoints and quotas" — AWS, accessed 2026-11-08. <https://docs.aws.amazon.com/general/latest/gr/bedrock.html> — cited in [Ch 19](../ch19/README.md#references)
- "Amazon Bedrock pricing" — AWS, accessed 2026-11-08. <https://aws.amazon.com/bedrock/pricing/> — cited in [Ch 19](../ch19/README.md#references)
- "Amazon SageMaker AI pricing" — AWS, accessed 2026-11-08. <https://aws.amazon.com/sagemaker/ai/pricing/> — cited in [Ch 19](../ch19/README.md#references)
- "Custom model hyperparameters" — AWS, accessed 2026-11-08. <https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models-hp.md> — cited in [Ch 19](../ch19/README.md#references)
- "Customize a model with fine-tuning in Amazon Bedrock" (supported models and ids) — AWS, accessed 2026-11-08. <https://docs.aws.amazon.com/bedrock/latest/userguide/custom-model-fine-tuning.html> — cited in [Ch 19](../ch19/README.md#references)
- "Customize Amazon Nova models with Amazon Bedrock fine-tuning" — AWS Machine Learning Blog. <https://aws.amazon.com/blogs/machine-learning/customize-amazon-nova-models-with-amazon-bedrock-fine-tuning/> — cited in [Ch 19](../ch19/README.md#references)
- "Deploy a custom model for on-demand inference" — AWS, accessed 2026-11-08. <https://docs.aws.amazon.com/bedrock/latest/userguide/deploy-custom-model-on-demand.md> — cited in [Ch 19](../ch19/README.md#references)
- "Evolving GitHub Copilot's next edit suggestions through custom model training" — GitHub Blog. <https://github.blog/ai-and-ml/github-copilot/evolving-github-copilots-next-edit-suggestions-through-custom-model-training/> — cited in [Ch 20](../ch20/README.md#references)
- "Fine-tune Amazon Nova models for accurate email data extraction" (Parcel Perform) — AWS Machine Learning Blog. <https://aws.amazon.com/blogs/machine-learning/fine-tune-amazon-nova-models-for-accurate-email-data-extraction/> — cited in [Ch 19](../ch19/README.md#references)
- "Fine-tuning" API reference — OpenAI, accessed 2026-11-08. <https://developers.openai.com/api/reference/resources/fine_tuning> — cited in [Ch 19](../ch19/README.md#references)
- "Fine-tuning best practices" — OpenAI, accessed 2026-11-08. <https://developers.openai.com/api/docs/guides/fine-tuning-best-practices> — cited in [Ch 19](../ch19/README.md#references)
- "How XY.AI Labs Built Customer-Specific EOB Parsers with Serverless Fine-Tuning" — Together AI, accessed 2026-11-08. <https://www.together.ai/customers/xy-ai-labs> — cited in [Ch 19](../ch19/README.md#references)
- "LoRA Adapters" — vLLM docs, accessed 2026-11-09. <https://docs.vllm.ai/en/latest/features/lora/> — cited in [Ch 18](../ch18/README.md#references)
- Zhao et al., "LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report" — arXiv, 2024-04-29. <https://arxiv.org/abs/2405.00732> — cited in [Ch 18](../ch18/README.md#references)
- Biderman et al., "LoRA Learns Less and Forgets Less" — TMLR, 2024. <https://arxiv.org/abs/2405.09673> — cited in [Ch 18](../ch18/README.md#references)
- "LoRA vs. full fine-tuning" (serverless LoRA discontinued) — Together AI docs, accessed 2026-11-08. <https://docs.together.ai/docs/fine-tuning/lora-vs-full> — cited in [Ch 19](../ch19/README.md#references)
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" — arXiv, 2021. <https://arxiv.org/abs/2106.09685> — cited in [Ch 18](../ch18/README.md#references)
- "Model lifecycle (Legacy)" — AWS, accessed 2026-11-08. <https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle-legacy.html> — cited in [Ch 19](../ch19/README.md#references)
- "Model versions and lifecycle" — Google Cloud, accessed 2026-11-08. <https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions> — cited in [Ch 19](../ch19/README.md#references)
- "Open weight model customization" — Amazon SageMaker AI, accessed 2026-11-08. <https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-open-weight.html> — cited in [Ch 19](../ch19/README.md#references)
- "Preparing data for multimodal fine-tuning" (Nova, `bedrock-conversation-2024`, reserved keywords) — AWS, accessed 2026-11-08. <https://docs.aws.amazon.com/nova/latest/userguide/fine-tune-prepare-data-understanding.html> — cited in [Ch 19](../ch19/README.md#references)
- "Pricing" (fine-tuning) — Together AI docs, accessed 2026-11-08. <https://docs.together.ai/docs/fine-tuning/pricing> — cited in [Ch 19](../ch19/README.md#references)
- "Pricing" — Fireworks AI, accessed 2026-11-08. <https://fireworks.ai/pricing> — cited in [Ch 19](../ch19/README.md#references)
- "Private Endpoints: Route Your Own Model Deployments Through OpenRouter" — OpenRouter, 2026-10-14. <https://openrouter.ai/blog/announcements/private-endpoints/> — cited in [Ch 19](../ch19/README.md#references)
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" — arXiv, 2023. <https://arxiv.org/abs/2305.14314> — cited in [Ch 18](../ch18/README.md#references)
- "Resolve errors when I fine-tune models on Amazon Bedrock" — AWS re:Post Knowledge Center, accessed 2026-11-08. <https://repost.aws/knowledge-center/bedrock-fine-tuning-errors> — cited in [Ch 19](../ch19/README.md#references)
- sagemaker 3.29.0, boto3 1.43.126, google-genai 3.0.0 — PyPI, accessed 2026-11-08. <https://pypi.org/project/sagemaker/> · <https://pypi.org/project/boto3/> · <https://pypi.org/project/google-genai/> — cited in [Ch 19](../ch19/README.md#references)
- SFT fine-tuning example notebook (SDK v3, serverless) — aws/sagemaker-python-sdk, accessed 2026-11-08. <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/model-customization-examples/serverless/sft_finetuning_example_notebook_pysdk_prod_v3.ipynb> — cited in [Ch 19](../ch19/README.md#references)
- "Streamlining Background Check Classification with Fine-tuned Small Language Models" (Checkr) — ZenML LLMOps Database, 2024. <https://www.zenml.io/llmops-database/streamlining-background-check-classification-with-fine-tuned-small-language-models> — cited in [Ch 18](../ch18/README.md#references)
- "Submit a model customization job for fine-tuning" — AWS, accessed 2026-11-08. <https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-submit.html> — cited in [Ch 19](../ch19/README.md#references)
- "Supervised fine-tuning" — OpenAI, accessed 2026-11-08. <https://developers.openai.com/api/docs/guides/supervised-fine-tuning> — cited in [Ch 19](../ch19/README.md#references)
- "Supervised Fine-Tuning - Text" — Fireworks AI docs, accessed 2026-11-08. <https://docs.fireworks.ai/fine-tuning/fine-tuning-models> — cited in [Ch 19](../ch19/README.md#references)
- TRL `SFTTrainer` — Hugging Face docs. <https://huggingface.co/docs/trl/sft_trainer> — cited in [Ch 18](../ch18/README.md#references)
- "Tune Gemini models with supervised fine-tuning" — Google Cloud, accessed 2026-11-08. <https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/supervised-tuning/use> — cited in [Ch 19](../ch19/README.md#references)
- "Tuning gen-AI? Here's the top 5 ways hundreds of orgs are doing it" (NextNet) — Google Cloud Blog, 2025-02-06. <https://cloud.google.com/transform/top-five-gen-ai-tuning-use-cases-gemini-hundreds-of-orgs> — cited in [Ch 19](../ch19/README.md#references)
- Unsloth — GitHub. <https://github.com/unslothai/unsloth> — cited in [Ch 18](../ch18/README.md#references)
- "Upload file" API reference (512 MB limit) — OpenAI, accessed 2026-11-08. <https://developers.openai.com/api/reference/resources/files/methods/create> — cited in [Ch 19](../ch19/README.md#references)

## Code-review bots & the GitHub API

- actions/checkout v7 and astral-sh/setup-uv v10 — GitHub, accessed 2026-11-07. <https://github.com/actions/checkout> · <https://github.com/astral-sh/setup-uv> — cited in [Ch 12](../ch12/README.md#references)
- API versions (2026-03-10, 2022-11-28) — GitHub Docs, accessed 2026-11-07. <https://docs.github.com/en/rest/about-the-rest-api/api-versions> — cited in [Ch 12](../ch12/README.md#references)
- "How to Make LLMs Shut Up" — Greptile blog, 2024-12-16. <https://www.greptile.com/blog/make-llms-shut-up> — cited in [Ch 12](../ch12/README.md#references)
- "Orchestrating AI Code Review at scale" — Cloudflare blog, 2026-04-20. <https://blog.cloudflare.com/ai-code-review> — cited in [Ch 12](../ch12/README.md#references)
- REST API endpoints for pull request reviews — GitHub Docs, accessed 2026-11-07. <https://docs.github.com/en/rest/pulls/reviews> — cited in [Ch 12](../ch12/README.md#references)
- "uReview: Scalable, Trustworthy GenAI for Code Review at Uber" — Uber Engineering, 2025-08-12. <https://www.uber.com/us/en/blog/ureview/> — cited in [Ch 12](../ch12/README.md#references)

## Deployment & scaling

- actionlint 1.7.12 — rhysd. <https://github.com/rhysd/actionlint> — cited in [Ch 20](../ch20/README.md#references)
- Claude API errors (529 `overloaded_error`) — Anthropic docs, accessed 2026-11-07. <https://platform.claude.com/docs/en/api/errors> — cited in [Ch 21](../ch21/README.md#references)
- Concurrency and async / await — FastAPI documentation. <https://fastapi.tiangolo.com/async/> — cited in [Ch 21](../ch21/README.md#references)
- Configuration (headless, `--csv`) — Locust documentation. <https://docs.locust.io/en/stable/configuration.html> — cited in [Ch 21](../ch21/README.md#references)
- Configure liveness, readiness and startup probes — Kubernetes docs. <https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/> — cited in [Ch 20](../ch20/README.md#references)
- "Connection idle timeout" — AWS Application Load Balancer docs. <https://docs.aws.amazon.com/elasticloadbalancing/latest/application/edit-load-balancer-attributes.html> — cited in [Ch 22 › F2](../ch22/README.md#f2-streaming-disconnects-behind-proxies)
- Connection Pools (`BlockingConnectionPool`) — redis-py documentation. <https://redis.readthedocs.io/en/stable/connections.html> — cited in [Ch 21](../ch21/README.md#references)
- Control the concurrency of workflows and jobs — GitHub Docs. <https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency> — cited in [Ch 20](../ch20/README.md#references)
- Deployments (rollout, undo, revision history) — Kubernetes docs. <https://kubernetes.io/docs/concepts/workloads/controllers/deployment/> — cited in [Ch 20](../ch20/README.md#references)
- Disruptions (PodDisruptionBudget) — Kubernetes docs. <https://kubernetes.io/docs/concepts/workloads/pods/disruptions/> — cited in [Ch 20](../ch20/README.md#references)
- Distributed load generation (`--processes`) — Locust documentation. <https://docs.locust.io/en/stable/running-distributed.html> — cited in [Ch 21](../ch21/README.md#references)
- "Exponential Backoff And Jitter" — AWS Architecture Blog (Marc Brooker), 2015-03-04. <https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/> — cited in [Ch 21](../ch21/README.md#references), [Ch 22 › F1](../ch22/README.md#f1-429-storms-synchronized-retries)
- "Graceful Kubernetes deployments" — Rakuten Engineering. <https://engineering.rakuten.today/post/graceful-k8s-delpoyments/> — cited in [Ch 20](../ch20/README.md#references)
- Horizontal Pod Autoscaling — Kubernetes docs, accessed 2026-11-07. <https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/> — cited in [Ch 20](../ch20/README.md#references)
- "Ingress NGINX retirement" — Kubernetes Blog, 2025-11-11. <https://kubernetes.io/blog/2025/11/11/ingress-nginx-retirement/> — cited in [Ch 20](../ch20/README.md#references)
- Install and set up kubectl on Linux — Kubernetes docs, accessed 2026-11-07. <https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/> — cited in [Ch 20](../ch20/README.md#references)
- Installing Helm — Helm docs (v4.3.1), accessed 2026-11-07. <https://helm.sh/docs/intro/install/> — cited in [Ch 20](../ch20/README.md#references)
- kind quick start (v0.33.0) — Kubernetes SIGs, accessed 2026-11-07. <https://kind.sigs.k8s.io/docs/user/quick-start/> — cited in [Ch 20](../ch20/README.md#references)
- "KiND – How I Wasted a Day Loading Local Docker Images" — iximiuz. <https://iximiuz.com/en/posts/kubernetes-kind-load-docker-image/> — cited in [Ch 20](../ch20/README.md#references)
- Little's law — Wikipedia. <https://en.wikipedia.org/wiki/Little%27s_law> — cited in [Ch 21](../ch21/README.md#references)
- "Liveness, Readiness, and Startup Probes" — Kubernetes docs. <https://kubernetes.io/docs/concepts/workloads/pods/probes/> — cited in [Ch 22 › F3](../ch22/README.md#f3-sync-health-check-starves-under-load)
- locust 2.47.0 — PyPI. <https://pypi.org/project/locust/> — cited in [Ch 21](../ch21/README.md#references)
- Managing environments for deployment — GitHub Docs. <https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments> — cited in [Ch 20](../ch20/README.md#references)
- metrics-server v0.9.1 — Kubernetes SIGs. <https://github.com/kubernetes-sigs/metrics-server> — cited in [Ch 20](../ch20/README.md#references)
- Multi-stage builds — Docker docs. <https://docs.docker.com/build/building/multi-stage/> — cited in [Ch 20](../ch20/README.md#references)
- openai-python README (retries, async client) — OpenAI, accessed 2026-11-07. <https://github.com/openai/openai-python> — cited in [Ch 21](../ch21/README.md#references)
- Rate limits — Anthropic Claude Platform docs, accessed 2026-11-07. <https://platform.claude.com/docs/en/api/rate-limits> — cited in [Ch 21](../ch21/README.md#references)
- "Rate limits" — OpenAI docs, accessed 2026-11-07. <https://developers.openai.com/api/docs/guides/rate-limits> — cited in [Ch 21](../ch21/README.md#references), [Ch 22 › F1](../ch22/README.md#f1-429-storms-synchronized-retries)
- "Scaling PostgreSQL to power 800 million ChatGPT users" — OpenAI (Bohan Zhang), 2026-01-22. <https://openai.com/index/scaling-postgresql/> — cited in [Ch 21](../ch21/README.md#references)
- "Scaling your API with rate limiters" — Stripe (Paul Tarjan), 2017-03-30. <https://stripe.com/blog/rate-limiters> — cited in [Ch 21](../ch21/README.md#references)
- Secrets — Kubernetes docs. <https://kubernetes.io/docs/concepts/configuration/secret/> — cited in [Ch 20](../ch20/README.md#references)
- SELECT: The Locking Clause (`SKIP LOCKED`) — PostgreSQL documentation (current). <https://www.postgresql.org/docs/current/sql-select.html> — cited in [Ch 21](../ch21/README.md#references)
- "Set, use, and manage variables in a Compose file with interpolation" — Docker docs. <https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/> — cited in [Ch 14](../ch14/README.md#references), [Ch 22 › F5](../ch22/README.md#f5-compose-env-precedence-and-var-on-down)
- Solid Queue README — Rails, accessed 2026-11-07. <https://github.com/rails/solid_queue> — cited in [Ch 21](../ch21/README.md#references)
- Use Docker service containers — GitHub Docs. <https://docs.github.com/en/actions/tutorials/use-containerized-services/use-docker-service-containers> — cited in [Ch 20](../ch20/README.md#references)
- Using profiles with Compose — Docker docs. <https://docs.docker.com/compose/how-tos/profiles/> — cited in [Ch 20](../ch20/README.md#references)
- Using Redis as a broker (visibility timeout) — Celery documentation (stable). <https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html> — cited in [Ch 21](../ch21/README.md#references)
- Using uv in Docker — Astral docs, accessed 2026-11-07. <https://docs.astral.sh/uv/guides/integration/docker/> — cited in [Ch 20](../ch20/README.md#references)
- Working with threads (default limiter of 40) — AnyIO documentation. <https://anyio.readthedocs.io/en/stable/threads.html> — cited in [Ch 21](../ch21/README.md#references)

## Dates, money & locales

- glibc locale data `localedata/locales/de_DE` (`am_pm "";""`) — sourceware.org. <https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=localedata/locales/de_DE;hb=HEAD> — cited in [Ch 22 › G3](../ch22/README.md#g3-strftimep-empty-under-a-non-english-locale)
- "Date/Time Functions and Operators" — PostgreSQL docs. <https://www.postgresql.org/docs/current/functions-datetime.html> — cited in [Ch 22 › G1](../ch22/README.md#g1-timezone-today)
- Paul Ganssle, "Semantics of timezone-aware datetime arithmetic" — 2018-02-22. <https://blog.ganssle.io/articles/2018/02/aware-datetime-arithmetic.html> — cited in [Ch 22 › G4](../ch22/README.md#g4-aware-datetime-arithmetic-ignores-dst)
- strftime(3) — Linux man-pages. <https://man7.org/linux/man-pages/man3/strftime.3.html> — cited in [Ch 22 › G3](../ch22/README.md#g3-strftimep-empty-under-a-non-english-locale)
- "Supported currencies" (minor units, zero-decimal currencies) — Stripe docs. <https://docs.stripe.com/currencies> — cited in [Ch 22 › G2](../ch22/README.md#g2-cents-vs-dollars)
- datetime "Supported operations" — Python docs. <https://docs.python.org/3/library/datetime.html> — cited in [Ch 22 › G4](../ch22/README.md#g4-aware-datetime-arithmetic-ignores-dst)
- The Subscription object (`status: canceled`) — Stripe API docs. <https://docs.stripe.com/api/subscriptions/object> — cited in [Ch 22 › G5](../ch22/README.md#g5-silent-no-op-filter-canceled-vs-cancelled)

## Incidents & postmortems

- "A Chevy dealership added an AI chatbot to its site. Then all hell broke loose." — Business Insider, 2023-12-18. <https://www.businessinsider.com/car-dealership-chevrolet-chatbot-chatgpt-pranks-chevy-2023-12> — cited in [Ch 1](../ch01/README.md#references)
- "A postmortem of three recent issues" — Anthropic Engineering, 2025-09-17. <https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues> — cited in [Ch 14](../ch14/README.md#references), [Ch 20](../ch20/README.md#references), [Ch 22 › J1](../ch22/README.md#j1-model-deprecation-and-silent-drift)
- "Air Canada found liable for chatbot's bad advice on plane tickets" (Moffatt v. Air Canada, 2024 BCCRT 149) — CBC News, 2024-02-15. <https://www.cbc.ca/news/canada/british-columbia/air-canada-chatbot-lawsuit-1.7116416> — cited in [Ch 1](../ch01/README.md#references), [Ch 22](../ch22/README.md#references)
- "An AI Customer Service Chatbot Made Up a Company Policy—and Created a Mess" — Benj Edwards, Ars Technica (via WIRED), 2025-04-19. <https://www.wired.com/story/cursor-ai-hallucination-policy-customer-service/> — cited in [Ch 5](../ch05/README.md#references)
- "API, ChatGPT & Sora Facing Issues" (incident write-up, December 11, 2024) — OpenAI Status. <https://status.openai.com/incidents/01JMYB483C404VMPCW726E8MET> — cited in [Ch 20](../ch20/README.md#references), [Ch 21](../ch21/README.md#references)
- "Car Dealership Disturbed When Its AI Is Caught Offering Chevys for $1 Each" — Futurism, 2023-12-20. <https://futurism.com/the-byte/car-dealership-ai> — cited in [Ch 13](../ch13/README.md#references), [Ch 22](../ch22/README.md#references)
- "Claude-powered AI agent's confession after deleting a firm's entire database" — The Guardian, 2026-04-29. <https://www.theguardian.com/technology/2026/apr/29/claude-ai-deletes-firm-database> — cited in [Ch 22](../ch22/README.md#references)
- "DPD error caused chatbot to swear at customer" — BBC News, 2024-01-19. <https://www.bbc.com/news/technology-68025677> — cited in [Ch 1](../ch01/README.md#references)
- "Expanding on what we missed with sycophancy" — OpenAI, 2025-05-02. <https://openai.com/index/expanding-on-sycophancy/> — cited in [Ch 14](../ch14/README.md#references), [Ch 20](../ch20/README.md#references)
- "OpenAI admits it screwed up testing its 'sychophant-y' ChatGPT update" — The Verge, 2025-05-05. <https://www.theverge.com/news/661422/openai-chatgpt-sycophancy-update-what-went-wrong> — cited in [Ch 3](../ch03/README.md#references)
- "Replit AI agent snafu 'shot across the bow' for vibe coding" — TechTarget, 2025-07-22. <https://www.techtarget.com/searchsoftwarequality/news/366627829/Replit-AI-agent-snafu-shot-across-the-bow-for-vibe-coding> — cited in [Ch 10](../ch10/README.md#references)
- "Vibe coding service Replit deleted user's production database, faked data, told fibs galore" — The Register, 2025-07-21. <https://www.theregister.com/software/2025/07/21/vibe-coding-service-replit-deleted-production-database/719783> — cited in [Ch 9](../ch09/README.md#references), [Ch 22](../ch22/README.md#references)

## Regulation

- "AI Omnibus enters into force" — European Commission, 2026-07-27. <https://digital-strategy.ec.europa.eu/en/news/ai-omnibus-enters-force> — cited in [Ch 22](../ch22/README.md#references)
- "Digital Omnibus: what is postponed in the EU AI Act (and what is not)" — AI Act Insight, updated 2026-09-04. <https://aiactinsight.eu/en/eu-ai-act/digital-omnibus/> — cited in [Ch 22](../ch22/README.md#references)
- Regulation (EU) 2026/1744 (Digital Omnibus on AI) — EUR-Lex. <https://eur-lex.europa.eu/eli/reg/2026/1744/oj/eng> — cited in [Ch 22](../ch22/README.md#references)
- "Timeline for the implementation of the EU AI Act" — European Commission AI Act Service Desk, accessed 2026-11-08. <https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act> — cited in [Ch 22](../ch22/README.md#references)

## Practitioner threads

Bug reports, forum threads and Q&A answers that the chapters cite as pain points. They record what practitioners hit at the time; check the thread's state before relying on it.

### GitHub issues, pull requests & discussions

- "_send_single_request logs sensitive data / leaks credentials" — httpx discussion #2765, 2023-07-08. <https://github.com/encode/httpx/discussions/2765> — cited in [Ch 22 › H1](../ch22/README.md#h1-pii-in-logs)
- vibrantlabsai/ragas #3028, "A row that errors leaves the denominator, so the reported score rises as the system degrades" — GitHub, 2026-09-23. <https://github.com/vibrantlabsai/ragas/issues/3028> — cited in [Ch 3](../ch03/README.md#references)
- "Agent loop non-termination: model retries identical failing tool calls indefinitely" — claude-code issue #30150, 2026-03-02. <https://github.com/anthropics/claude-code/issues/30150> — cited in [Ch 22 › D3](../ch22/README.md#d3-agent-cost-runaway-loop)
- "Agent-level retry creates duplicate charges — no idempotency guard above the tool layer" — stripe/ai issue #402, 2026-05-03. <https://github.com/stripe/ai/issues/402> — cited in [Ch 22 › D2](../ch22/README.md#d2-retry--duplicate-side-effects)
- "AI Chat: per-second timestamp in system prompt defeats OpenAI/Copilot prefix caching" — dbeaver issue #41321, 2026-06-16. <https://github.com/dbeaver/dbeaver/issues/41321> — cited in [Ch 22 › I1](../ch22/README.md#i1-prompt-cache-invalidation)
- "Anthropic prompt caching broken - Cache Read always 0" — openclaw issue #19534, 2026-02-17. <https://github.com/openclaw/openclaw/issues/19534> — cited in [Ch 22 › I1](../ch22/README.md#i1-prompt-cache-invalidation)
- "/api/embed uses 512 token context window even though model was configured with 8192" — ollama/ollama issue #7008, 2024-09-27. <https://github.com/ollama/ollama/issues/7008> — cited in [Ch 4](../ch04/README.md#references)
- "At debug log level API requests to OpenAI get logged with all headers, including an API key" — openai-python issue #1196, 2024-02-26 (closed 2026-09-15). <https://github.com/openai/openai-python/issues/1196> — cited in [Ch 22 › H1](../ch22/README.md#h1-pii-in-logs)
- openai/openai-python #3777, "Blocking the event loop since 3.x ?", 2026-09-01. <https://github.com/openai/openai-python/issues/3777> — cited in [Ch 21](../ch21/README.md#references)
- "Buffering breaks event streams" — nginx issue #1505, 2026-06-24. <https://github.com/nginx/nginx/issues/1505> — cited in [Ch 22 › F2](../ch22/README.md#f2-streaming-disconnects-behind-proxies)
- "bug: ClickHouse system tables grow unbounded in default docker-compose deployment" — langfuse/langfuse issue #13123, GitHub. <https://github.com/langfuse/langfuse/issues/13123> — cited in [Ch 14](../ch14/README.md#references)
- "bug: unbounded ClickHouse Docker log can exhaust self-hosted server after ENOSPC" — langfuse/langfuse issue #16339, GitHub. <https://github.com/langfuse/langfuse/issues/16339> — cited in [Ch 14](../ch14/README.md#references)
- unclecode/crawl4ai #1663, "Claude Sonnet returns markdown-wrapped JSON despite JSON mode being enabled", 2025-12-09. <https://github.com/unclecode/crawl4ai/issues/1663> — cited in [Ch 2](../ch02/README.md#references), [Ch 22 › B1](../ch22/README.md#b1-json_object-returns-fenced-json)
- "Consistency problem with --env-file usage at least on up and down commands" — docker/compose issue #8515, 2021-07-15. <https://github.com/docker/compose/issues/8515> — cited in [Ch 22 › F5](../ch22/README.md#f5-compose-env-precedence-and-var-on-down)
- ggml-org/llama.cpp #11681, "Context size is being divided by number of parallel inferences". <https://github.com/ggml-org/llama.cpp/issues/11681> — cited in [Ch 16](../ch16/README.md#references)
- "Counter with labels not initialized" — prometheus/client_java issue #632, 2021-03-05. <https://github.com/prometheus/client_java/issues/632> — cited in [Ch 22 › J2](../ch22/README.md#j2-labeled-prometheus-counter-missing-until-first-increment)
- "CrashLoopBackOff in air-gapped OpenShift due to tiktoken trying to download cl100k_base.tiktoken" — LiteLLM issue #23218, 2026-03-10. <https://github.com/BerriAI/litellm/issues/23218> — cited in [Ch 22 › F4](../ch22/README.md#f4-tiktoken-downloads-at-runtime-offline-crash)
- redis/redis-py #4458, "Document BlockingConnectionPool (easy to hit "Too many connections" since 8.0)", 2026-10-19. <https://github.com/redis/redis-py/issues/4458> — cited in [Ch 21](../ch21/README.md#references)
- earendil-works/pi issue #1873 (a timestamp makes every system prompt unique) — GitHub, 2026-03-06. <https://github.com/earendil-works/pi/issues/1873> — cited in [Ch 17](../ch17/README.md#references)
- "entry too far behind, oldest acceptable timestamp is..." — grafana/loki issue #9869, GitHub, 2023 (closed 2026-09-25). <https://github.com/grafana/loki/issues/9869> — cited in [Ch 11](../ch11/README.md#references)
- "EPIC: Concurrency & async-isolation hardening" — titiler-openeo issue #304, 2026-06-22. <https://github.com/sentinel-hub/titiler-openeo/issues/304> — cited in [Ch 22 › F3](../ch22/README.md#f3-sync-health-check-starves-under-load)
- home-assistant/core #150601, "Error talking to OpenAI with gpt-5 (and other new) model". <https://github.com/home-assistant/core/issues/150601> — cited in [Ch 1](../ch01/README.md#references)
- "Excessive context usage for tools" — github-mcp-server issue #1286, 2025-10-23. <https://github.com/github/github-mcp-server/issues/1286> — cited in [Ch 22 › I2](../ch22/README.md#i2-tool-schemas-tax-every-call)
- fastapi/fastapi discussion #15794, anti-buffering headers on streaming responses. <https://github.com/fastapi/fastapi/discussions/15794> — cited in [Ch 1](../ch01/README.md#references)
- "Feature request: Configurable PostgreSQL schema for langgraph-checkpoint-postgres" — GitHub issue #7345, 2026-03-30. <https://github.com/langchain-ai/langgraph/issues/7345> — cited in [Ch 10](../ch10/README.md#references)
- "[FEATURE] Detect and stop repeated tool-call loops" — strands-agents/harness-sdk issue #5629, GitHub, 2026. <https://github.com/strands-agents/harness-sdk/issues/5629> — cited in [Ch 9](../ch09/README.md#references)
- "[Feature] Support TimeZone-aware time dimension types" (issue #733) — dbt-labs/metricflow, GitHub, opened 2023-08-21. <https://github.com/dbt-labs/metricflow/issues/733> — cited in [Ch 7](../ch07/README.md#references)
- "Filtered pgvector HNSW conclusion queries silently under-return with iterative scans off" — plastic-labs/honcho issue #1048, 2026-08-23, and fix PR #1060. <https://github.com/plastic-labs/honcho/issues/1048> · <https://github.com/plastic-labs/honcho/pull/1060> — cited in [Ch 4](../ch04/README.md#references)
- "Functionality to adjust exponential backoff associated with max_retries option" — openai-python issue #1102. <https://github.com/openai/openai-python/issues/1102> — cited in [Ch 22 › F1](../ch22/README.md#f1-429-storms-synchronized-retries)
- "GET /companies/:id/heartbeat-runs silently ignores status and invocationSource filters" — paperclip issue #4185, 2026-04-21. <https://github.com/paperclipai/paperclip/issues/4185> — cited in [Ch 22 › G5](../ch22/README.md#g5-silent-no-op-filter-canceled-vs-cancelled)
- "Getting content_filter as finish_reason … without prompting any sensitive content" — LangChain issue #11069, 2023-09-26. <https://github.com/langchain-ai/langchain/issues/11069> — cited in [Ch 22 › B3](../ch22/README.md#b3-safety-filter-empty-reply-finish_reasoncontent_filter)
- "Getting fewer results after indexing using HNSW" — pgvector issue #684, 2024-09-27. <https://github.com/pgvector/pgvector/issues/684> — cited in [Ch 22 › C1](../ch22/README.md#c1-filtered-hnsw-query-returns-fewer-rows-than-limit)
- "glm-ocr cannot process images > 2048x2048" — ollama/ollama #14114, 2026-02-06. <https://github.com/ollama/ollama/issues/14114> — cited in [Ch 8](../ch08/README.md#references)
- "glm-ocr crashes with default num_ctx" — ollama/ollama #16696, 2026-06-12. <https://github.com/ollama/ollama/issues/16696> — cited in [Ch 8](../ch08/README.md#references)
- "glm-ocr Table Recognition regression 0.34.0 → 0.35.1" — ollama/ollama #18810, 2026-10-05. <https://github.com/ollama/ollama/issues/18810> — cited in [Ch 8](../ch08/README.md#references)
- "Guardrails is very slow. How to improve speed?" — GitHub discussion #587, NVIDIA-NeMo/Guardrails, 2024-06-28. <https://github.com/NVIDIA-NeMo/Guardrails/discussions/587> — cited in [Ch 13](../ch13/README.md#references)
- "Handle JSON responses wrapped in Markdown code fences in AI Agent Task" — camunda/connectors issue #6275, 2026-02-05. <https://github.com/camunda/connectors/issues/6275> — cited in [Ch 22 › B1](../ch22/README.md#b1-json_object-returns-fenced-json)
- "How do I keep data in Postgres checkpointer database from growing unbounded?" — GitHub issue langchain-ai/langgraphjs#1138, 2025-04-29. <https://github.com/langchain-ai/langgraphjs/issues/1138> — cited in [Ch 5](../ch05/README.md#references), [Ch 10](../ch10/README.md#references)
- "How to inject fixed parameters into MCP tool calls to prevent LLM hallucination?" — agno discussion #7410, 2026-04-08. <https://github.com/agno-agi/agno/discussions/7410> — cited in [Ch 22 › D1](../ch22/README.md#d1-tool-argument-hallucination)
- "How to limit the max number of threads with sync endpoints?" — FastAPI discussion #8690, 2021-11-29. <https://github.com/fastapi/fastapi/discussions/8690> — cited in [Ch 22 › F3](../ch22/README.md#f3-sync-health-check-starves-under-load)
- openai/openai-python #1596, "httpx client has very poor performance for concurrent requests compared to aiohttp", 2024-08-05. <https://github.com/openai/openai-python/issues/1596> — cited in [Ch 21](../ch21/README.md#references)
- "inconsistent usage of 'canceled' vs 'cancelled'" — Linear issue #1116, 2026-04-22. <https://github.com/linear/linear/issues/1116> — cited in [Ch 22 › G5](../ch22/README.md#g5-silent-no-op-filter-canceled-vs-cancelled)
- "Is there a way for tiktoken to interoperate better with offline AI software?" — tiktoken issue #232, 2023-12-27. <https://github.com/openai/tiktoken/issues/232> — cited in [Ch 22 › F4](../ch22/README.md#f4-tiktoken-downloads-at-runtime-offline-crash)
- "Key max_budget not enforced" — BerriAI/litellm issue #26672, 2026-04-28. <https://github.com/BerriAI/litellm/issues/26672> — cited in [Ch 15](../ch15/README.md#references)
- kserve/kserve #5062, liveness probe kills large models before they finish loading. <https://github.com/kserve/kserve/issues/5062> — cited in [Ch 20](../ch20/README.md#references)
- kubernetes/ingress-nginx #14090, SSE streams closed after an nginx reload. <https://github.com/kubernetes/ingress-nginx/issues/14090> — cited in [Ch 20](../ch20/README.md#references)
- kubernetes/kubernetes #105703, existing connections survive endpoint removal. <https://github.com/kubernetes/kubernetes/issues/105703> — cited in [Ch 20](../ch20/README.md#references)
- "LangGraph [1.0.6] Bug - Agent infinite looping until recursion limit error is hit" — GitHub issue langchain-ai/langgraph#6731, 2026-01-30. <https://github.com/langchain-ai/langgraph/issues/6731> — cited in [Ch 5](../ch05/README.md#references), [Ch 22 › D3](../ch22/README.md#d3-agent-cost-runaway-loop)
- "Long tool calls (~180s+) silently re-executed from checkpoint on LangGraph Cloud" — LangGraph issue #7417, 2026-04-05. <https://github.com/langchain-ai/langgraph/issues/7417> — cited in [Ch 22 › D2](../ch22/README.md#d2-retry--duplicate-side-effects)
- "LoRA adapters trained on a 4-bit base degrade when served on a BF16 base" — huggingface/trl discussion #7231, GitHub, 2026-09-16. <https://github.com/huggingface/trl/discussions/7231> — cited in [Ch 18](../ch18/README.md#references)
- "`merge_and_unload` for a quantized model ruins its quality" — huggingface/transformers issue #31293, GitHub, 2024-06-06. <https://github.com/huggingface/transformers/issues/31293> — cited in [Ch 18](../ch18/README.md#references)
- vllm-project/vllm #5274, OOM vs "No available memory for the cache blocks" with `gpu_memory_utilization`. <https://github.com/vllm-project/vllm/issues/5274> — cited in [Ch 16](../ch16/README.md#references)
- vllm-project/vllm issue #2248 ("No available memory for the cache blocks") — GitHub, 2023-12-24. <https://github.com/vllm-project/vllm/issues/2248> — cited in [Ch 17](../ch17/README.md#references)
- "No safeguard against repeated identical tool calls" — PrimeIntellect-ai/prime-agent issue #1326, GitHub, 2026. <https://github.com/PrimeIntellect-ai/prime-agent/issues/1326> — cited in [Ch 9](../ch09/README.md#references)
- "num_ctx prompt truncation limit is silently half the configured value" — Ollama issue #17427, 2026-07-27. <https://github.com/ollama/ollama/issues/17427> — cited in [Ch 22 › A6](../ch22/README.md#a6-context-overflow-is-silent)
- ollama/ollama #10929, "Ollama produces invalid JSON when using thinking mode with structured output". <https://github.com/ollama/ollama/issues/10929> — cited in [Ch 2](../ch02/README.md#references)
- "ollama run silently truncating prompt" — Ollama issue #8099, 2024-12-14. <https://github.com/ollama/ollama/issues/8099> — cited in [Ch 22 › A6](../ch22/README.md#a6-context-overflow-is-silent)
- ollama/ollama #9054, parallel requests and queueing. <https://github.com/ollama/ollama/issues/9054> — cited in [Ch 16](../ch16/README.md#references)
- "OpenAI reasoning models (o1/o3/gpt-5) do not support temperature and other sampling parameters" — LibreChat issue #10737, 2025-12-01. <https://github.com/danny-avila/LibreChat/issues/10737> — cited in [Ch 22 › A4](../ch22/README.md#a4-reasoning-models-reject-or-silently-drop-temperature)
- "OpenAI representation fails to produce output when response content is None" — BERTopic issue #2176, 2024-10-11. <https://github.com/MaartenGr/BERTopic/issues/2176> — cited in [Ch 22 › A3](../ch22/README.md#a3-messagecontent-is-none)
- "Opus 4.7 safety filter produces false-positive refusals on encoded input" — claude-code issue #52809, 2026-04-24. <https://github.com/anthropics/claude-code/issues/52809> — cited in [Ch 22 › B3](../ch22/README.md#b3-safety-filter-empty-reply-finish_reasoncontent_filter)
- encode/httpx #1171, "PoolTimeout when num tasks in asyncio.gather() exceeds client max_connections", 2020-08-11. <https://github.com/encode/httpx/issues/1171> — cited in [Ch 21](../ch21/README.md#references)
- "Procedural memory creation crashes with AttributeError … when the LLM returns no content" — mem0 issue #6150, 2026-07-08. <https://github.com/mem0ai/mem0/issues/6150> — cited in [Ch 22 › A3](../ch22/README.md#a3-messagecontent-is-none)
- "Prompt Injection on LlamaGuard is not considered a vulnerability" — GitHub issue #77, meta-llama/PurpleLlama, 2025-02-21. <https://github.com/meta-llama/PurpleLlama/issues/77> — cited in [Ch 13](../ch13/README.md#references)
- pypdf issues #4130 (2026-09-26) and #4254 (2026-10-25) — GitHub. <https://github.com/py-pdf/pypdf/issues/4130> · <https://github.com/py-pdf/pypdf/issues/4254> — cited in [Ch 8](../ch08/README.md#references)
- "[Question] Why counters starts from NO_DATA rather than 0" — prometheus/prometheus issue #10466, GitHub. <https://github.com/prometheus/prometheus/issues/10466> — cited in [Ch 11](../ch11/README.md#references)
- QwenLM/Qwen3 #1286, disabling thinking under `vllm serve`. <https://github.com/QwenLM/Qwen3/issues/1286> — cited in [Ch 16](../ch16/README.md#references)
- "Review submission fails with 'Line could not be resolved' when inline comment targets line outside diff hunk" — pulumi-labs/gh-aw-internal issue #5, GitHub, 2026-04-06. <https://github.com/pulumi-labs/gh-aw-internal/issues/5> — cited in [Ch 12](../ch12/README.md#references)
- "SELECT WHERE ORDER BY LIMIT no results" — pgvector issue #719, 2024-11-26. <https://github.com/pgvector/pgvector/issues/719> — cited in [Ch 22 › C1](../ch22/README.md#c1-filtered-hnsw-query-returns-fewer-rows-than-limit)
- "Semantic cache replays stale answers in agent loops" (`max_messages`) — BerriAI/litellm PR #43878, 2026-09-30. <https://github.com/BerriAI/litellm/pull/43878> — cited in [Ch 15](../ch15/README.md#references)
- "Sentence Transformers Gets Stuck loading" — huggingface/transformers issue #30990, 2024-05-23. <https://github.com/huggingface/transformers/issues/30990> — cited in [Ch 4](../ch04/README.md#references)
- sgl-project/sglang #4259, NCCL shared memory errors in Docker. <https://github.com/sgl-project/sglang/issues/4259> — cited in [Ch 16](../ch16/README.md#references)
- "SSE streaming responses emit zero bytes during a long time-to-first-token" — LiteLLM issue #34819, 2026-07-27. <https://github.com/BerriAI/litellm/issues/34819> — cited in [Ch 22 › F2](../ch22/README.md#f2-streaming-disconnects-behind-proxies)
- "Stale bundled timezone data produces wrong Canadian date-range bounds after Nov 1, 2026" (issue #12431) — cube-js/cube, GitHub, 2026-10-26. <https://github.com/cube-js/cube/issues/12431> — cited in [Ch 7](../ch07/README.md#references)
- "stdio transport: log output on stdout breaks MCP JSON-RPC" — dirmacs/daedra issue #4, GitHub, 2026-01-29. <https://github.com/dirmacs/daedra/issues/4> — cited in [Ch 9](../ch09/README.md#references)
- "Structured output with OpenAI SDK and gpt-oss:20b not working" — ollama/ollama issue #11691, GitHub, 2025-08-05. <https://github.com/ollama/ollama/issues/11691> — cited in [Ch 2](../ch02/README.md#references), [Ch 12](../ch12/README.md#references)
- "Support for local time dimensions without query timezone conversion" (issue #10166) — cube-js/cube, GitHub, opened 2025-11-21. <https://github.com/cube-js/cube/issues/10166> — cited in [Ch 7](../ch07/README.md#references)
- openai/openai-python #2104, "temperature is not supported with this model (o3-mini)". <https://github.com/openai/openai-python/issues/2104> — cited in [Ch 1](../ch01/README.md#references)
- ollama/ollama #14850, "`think=False` breaks JSON structured output with `format` when using `qwen3.5:27b`", 2026-03-14. <https://github.com/ollama/ollama/issues/14850> — cited in [Ch 2](../ch02/README.md#references)
- "Time validation not working properly for languages does not have AM/PM" — dayjs issue #2442, 2023-09-07. <https://github.com/iamkun/dayjs/issues/2442> — cited in [Ch 22 › G3](../ch22/README.md#g3-strftimep-empty-under-a-non-english-locale)
- "Tool Calling Running in Loop Until Max-Turn" — openai/openai-agents-python issue #191, GitHub. <https://github.com/openai/openai-agents-python/issues/191> — cited in [Ch 9](../ch09/README.md#references)
- "Unable to finetune model following the official guide" — googleapis/python-genai issue #1324. <https://github.com/googleapis/python-genai/issues/1324> — cited in [Ch 19](../ch19/README.md#references)
- "Uvicorn failing to reply to healthchecks on K8s" — uvicorn discussion #2077, 2023-08-21. <https://github.com/Kludex/uvicorn/discussions/2077> — cited in [Ch 22 › F3](../ch22/README.md#f3-sync-health-check-starves-under-load)
- "[v3] FrameworkProcessor and ModelTrainer: 4 regressions ... from v2 migration" — aws/sagemaker-python-sdk issue #5765, 2026-04. <https://github.com/aws/sagemaker-python-sdk/issues/5765> — cited in [Ch 19](../ch19/README.md#references)
- vllm-project/vllm #58544, Helm chart liveness probe restarts pods during model load. <https://github.com/vllm-project/vllm/issues/58544> — cited in [Ch 20](../ch20/README.md#references)
- vllm-project/vllm issue #15025 (draft model about 30% slower than the target alone) — GitHub, 2025-03-18. <https://github.com/vllm-project/vllm/issues/15025> — cited in [Ch 17](../ch17/README.md#references)
- vllm-project/vllm issue #20112 (forcing `--quantization awq` skips `awq_marlin`) — GitHub, 2025-06-26. <https://github.com/vllm-project/vllm/issues/20112> — cited in [Ch 17](../ch17/README.md#references)
- vllm-project/vllm issue #42390 (FP8 on an RTX 3090 falls back to Marlin W8A16) — GitHub, 2026-05-12. <https://github.com/vllm-project/vllm/issues/42390> — cited in [Ch 17](../ch17/README.md#references)
- confident-ai/deepeval #3110, "what is a test case's canonical verdict when a judge-based metric is re-run and straddles the threshold?" — GitHub, 2026-08-25. <https://github.com/confident-ai/deepeval/issues/3110> — cited in [Ch 3](../ch03/README.md#references)

### Hacker News

- "A statistical approach to model evaluations" — Hacker News, 2024-11-23. <https://news.ycombinator.com/item?id=42220573> — cited in [Ch 22 › E3](../ch22/README.md#e3-small-eval-gates-are-noisy)
- "AgentShield – Open benchmark of 6 AI agent security tools (537 test cases)" — Hacker News, 2026-02-15. <https://news.ycombinator.com/item?id=47025096> — cited in [Ch 13](../ch13/README.md#references)
- "AI handles incidents, engineers lose touch with their systems" — Hacker News discussion, 2026-09-05. <https://news.ycombinator.com/item?id=49574167> — cited in [Ch 11](../ch11/README.md#references)
- "Any text-to-SQL benchmark should address difficulties of real-world data stores" — Hacker News, 2026. <https://news.ycombinator.com/item?id=49013995> — cited in [Ch 6](../ch06/README.md#references)
- "Ask HN: How are companies building LLM to SQL database features in production?" — Hacker News, 2025-02-11. <https://news.ycombinator.com/item?id=43015167> — cited in [Ch 6](../ch06/README.md#references)
- "Ask HN: How do you add guard rails in LLM response without breaking streaming?" — Hacker News, 2024-10-15. <https://news.ycombinator.com/item?id=41845256> — cited in [Ch 13](../ch13/README.md#references)
- "Build a search engine, not a vector DB" — Hacker News discussion, 2023-12-20. <https://news.ycombinator.com/item?id=38703943> — cited in [Ch 4](../ch04/README.md#references)
- Comment "Can someone explain this 'semantic layer' business (cube.dev)?" on "Building an open data pipeline in 2024" — Hacker News, 2024-04-27. <https://news.ycombinator.com/item?id=40182301> — cited in [Ch 7](../ch07/README.md#references)
- "Choose Postgres queue technology" — Hacker News discussion, 2023-09-24. <https://news.ycombinator.com/item?id=37636841> — cited in [Ch 21](../ch21/README.md#references)
- "Claude Token Counter, now with model comparisons" — Hacker News, 2026-04-20. <https://news.ycombinator.com/item?id=47829178> — cited in [Ch 22 › A5](../ch22/README.md#a5-unicode-and-tokenization-surprises)
- Comment on LangGraph state typing — Hacker News, 2025-06-18. <https://news.ycombinator.com/item?id=44307883> — cited in [Ch 10](../ch10/README.md#references)
- Comment thread on LiteLLM quality and budget enforcement — Hacker News, 2025-11-19. <https://news.ycombinator.com/item?id=45985220> — cited in [Ch 15](../ch15/README.md#references)
- "Cursor IDE support hallucinates lockout policy, causes user cancellations" — Hacker News, 2025-04-14. <https://news.ycombinator.com/item?id=43683012> — cited in [Ch 22](../ch22/README.md#references)
- "Data Exfiltration from Slack AI via indirect prompt injection" — Hacker News, 2024-08-20. <https://news.ycombinator.com/item?id=41302597> — cited in [Ch 13](../ch13/README.md#references), [Ch 22 › H2](../ch22/README.md#h2-prompt-injection-via-retrieved-content)
- "Dbt to acquire Transform to build out its semantic layer" — Hacker News, 2023-02-09. <https://news.ycombinator.com/item?id=34724207> — cited in [Ch 7](../ch07/README.md#references)
- Comment by muzani in "Fine-tuning LLMs is a waste of time" — Hacker News, 2025-06. <https://news.ycombinator.com/item?id=44242737> — cited in [Ch 19](../ch19/README.md#references)
- "How is ChatGPT's behavior changing over time?" — Hacker News, 2023-07-19. <https://news.ycombinator.com/item?id=36781015> — cited in [Ch 22 › J1](../ch22/README.md#j1-model-deprecation-and-silent-drift)
- "How we made our AI code review bot stop leaving nitpicky comments" — Hacker News discussion, 2024-12. <https://news.ycombinator.com/item?id=42451968> — cited in [Ch 12](../ch12/README.md#references)
- resiros, comment on "Launch HN: Traceloop" — Hacker News, 2024-07-17. <https://news.ycombinator.com/item?id=40987261> — cited in [Ch 3](../ch03/README.md#references)
- "LiteLLM PyPI package compromised" discussion — Hacker News, 2026-03-24. <https://news.ycombinator.com/item?id=47501426> — cited in [Ch 15](../ch15/README.md#references)
- "LLM Observability in the Wild – Why OpenTelemetry Should Be the Standard" (discussion of Phoenix span kinds showing as `unknown`) — Hacker News, 2025-09-27. <https://news.ycombinator.com/item?id=45398467> — cited in [Ch 14](../ch14/README.md#references)
- Comment on "Nanonets-OCR-s" — Hacker News, 2025-06-16. <https://news.ycombinator.com/item?id=44290651> — cited in [Ch 8](../ch08/README.md#references)
- "Positional preferences, order effects, prompt sensitivity undermine AI judgments" — Hacker News, 2025-05-23. <https://news.ycombinator.com/item?id=44074668> — cited in [Ch 22 › E1](../ch22/README.md#e1-llm-judge-position-bias)
- "Production RAG: what I learned from processing 5M+ documents" — Hacker News, 2025-10-20. <https://news.ycombinator.com/item?id=45645349> — cited in [Ch 4](../ch04/README.md#references)
- "The hate for MCP here is absurd..." (thread) — Hacker News, 2025-11-17. <https://news.ycombinator.com/item?id=45950814> — cited in [Ch 9](../ch09/README.md#references)
- "Using LLM for confidence scoring is a bad idea" — Hacker News, 2026-10-21. <https://news.ycombinator.com/item?id=50176082> — cited in [Ch 22 › E2](../ch22/README.md#e2-self-reported-confidence-is-uncalibrated)
- "We chose LangGraph to build our coding agent" — Hacker News discussion, 2025-03-25. <https://news.ycombinator.com/item?id=43468435> — cited in [Ch 5](../ch05/README.md#references)
- "Why is GPT-3 15.77x more expensive for certain languages?" — Hacker News, 2023-04-10. <https://news.ycombinator.com/item?id=35514623> — cited in [Ch 22 › A5](../ch22/README.md#a5-unicode-and-tokenization-surprises)

### Stack Overflow

- "Different date formats in same column" — Stack Overflow, 2025-10-01. <https://stackoverflow.com/questions/79780171/different-date-formats-in-same-column> — cited in [Ch 8](../ch08/README.md#references)
- "Env variable invalid interpolation format for 'required variable is missing a value'" — Stack Overflow, 2022-03-12. <https://stackoverflow.com/questions/71453411/env-variable-invalid-interpolation-format-for-required-variable-is-missing-a-va> — cited in [Ch 22 › F5](../ch22/README.md#f5-compose-env-precedence-and-var-on-down)
- "EventSource / Server-Sent Events through Nginx" — Stack Overflow, 2012-12-02. <https://stackoverflow.com/questions/13672743/eventsource-server-sent-events-through-nginx> — cited in [Ch 22 › F2](../ch22/README.md#f2-streaming-disconnects-behind-proxies)
- "FastAPI runs API calls in serial instead of parallel fashion" — Stack Overflow, 2022-03-17. <https://stackoverflow.com/questions/71516140> — cited in [Ch 21](../ch21/README.md#references)
- "how to use tiktoken in offline mode computer" — Stack Overflow, 2023-04-26. <https://stackoverflow.com/questions/76106366/how-to-use-tiktoken-in-offline-mode-computer> — cited in [Ch 22 › F4](../ch22/README.md#f4-tiktoken-downloads-at-runtime-offline-crash)
- "Local start and end of day in UTC" — Stack Overflow, 2023-10-28. <https://stackoverflow.com/questions/77381592/local-start-and-end-of-day-in-utc> — cited in [Ch 22 › G1](../ch22/README.md#g1-timezone-today)
- "Matching only some words in a Postgres full text search" — Stack Overflow, 2016-02-14. <https://stackoverflow.com/questions/35391363/matching-only-some-words-in-a-postgres-full-text-search> — cited in [Ch 22 › C2](../ch22/README.md#c2-websearch_to_tsquery-ands-every-word-zero-hits)
- "PDF to text in python returning empty results in image files" — Stack Overflow (answer by the pypdf maintainer, 2023-06-18). <https://stackoverflow.com/questions/66283836/pdf-to-text-in-python-returning-empty-results-in-image-files> — cited in [Ch 8](../ch08/README.md#references)
- "Why does gpt-4-0613 hallucinate (make up) function parameters?" — Stack Overflow, 2023-07-03. <https://stackoverflow.com/questions/76608259/openai-gpt-4-api-why-does-gpt-4-0613-hallucinate-make-up-function-parameters> — cited in [Ch 22 › D1](../ch22/README.md#d1-tool-argument-hallucination)
- "Why not use Double or Float to represent currency?" — Stack Overflow, 2010-09-16. <https://stackoverflow.com/questions/3730019/why-not-use-double-or-float-to-represent-currency> — cited in [Ch 22 › G2](../ch22/README.md#g2-cents-vs-dollars)

### Forums, Q&A sites & community posts

- "Bedrock - Fine-Tuned Inference Bug" — AWS re:Post, accessed 2026-11-08. <https://repost.aws/questions/QUlMCRME6TRkK6XQoBSjP4pw/bedrock-fine-tuned-inference-bug> — cited in [Ch 19](../ch19/README.md#references)
- "Dataset flagged for sexual content" — OpenAI Developer Community, 2025-07. <https://community.openai.com/t/dataset-flagged-for-sexual-content/1310016> — cited in [Ch 19](../ch19/README.md#references)
- "Difference between two datetimes does not account for DST" — Python bug tracker #878424, 2004. <https://bugs.python.org/issue878424> — cited in [Ch 22 › G4](../ch22/README.md#g4-aware-datetime-arithmetic-ignores-dst)
- "Dreaded 429 rate limit errors when our usage is well-under the limits" — OpenAI Developer Community, 2023-05-13. <https://community.openai.com/t/dreaded-429-rate-limit-errors-when-our-usage-is-well-under-the-limits/207653> — cited in [Ch 21](../ch21/README.md#references)
- "Fine-tuning is stuck in validating files" — OpenAI Developer Community, 2026-10/11. <https://community.openai.com/t/fine-tuning-is-stuck-in-validating-files/1406992> — cited in [Ch 19](../ch19/README.md#references)
- "Fine-Tuning Queue/Jobs Taking Several Hours to Start With Gemini/Vertex AI" — Google Developer forums. <https://discuss.google.dev/t/fine-tuning-queue-jobs-taking-several-hours-to-start-with-gemini-vertex-ai/180327> — cited in [Ch 19](../ch19/README.md#references)
- "How are you handling runaway agent costs?" — OpenAI Community, 2026-06-13. <https://community.openai.com/t/how-are-you-handling-runaway-agent-costs/1383593> — cited in [Ch 22 › D3](../ch22/README.md#d3-agent-cost-runaway-loop)
- "nginx proxy_buffering Broke My Claude SSE Stream: 0.7s to 3.4s TTFT" — DEV Community. <https://dev.to/ji_ai/nginx-proxybuffering-broke-my-claude-sse-stream-07s-to-34s-ttft-30d2> — cited in [Ch 1](../ch01/README.md#references)
- "Nova Pro finetune job keeps stuck" — AWS re:Post, accessed 2026-11-08. <https://repost.aws/questions/QU71n1MTIGQNakWhIWelgQOg/nova-pro-finetune-job-keeps-stuck> — cited in [Ch 19](../ch19/README.md#references)
- "O1-mini model output '' with finish_reason of length" — OpenAI Community, 2024-11-12. <https://community.openai.com/t/o1-mini-model-output-with-finish-reason-of-length/1014258> — cited in [Ch 22 › A2](../ch22/README.md#a2-hidden-reasoning-eats-max_tokens-empty-answer-still-billed)
- "O4-mini returns empty response because reasoning token used all the completion token" — OpenAI Community, 2025-09-19. <https://community.openai.com/t/o4-mini-returns-empty-response-because-reasoning-token-used-all-the-completion-token/1359002> — cited in [Ch 22 › A2](../ch22/README.md#a2-hidden-reasoning-eats-max_tokens-empty-answer-still-billed)
- "Pre-interrupt() code re-runs on resume — anti-pattern?" — LangChain Forum, 2026-05-27. <https://forum.langchain.com/t/pre-interrupt-code-re-runs-on-resume-anti-pattern-or-is-there-a-sanctioned-way-to-detect-resume/3809> — cited in [Ch 22 › D4](../ch22/README.md#d4-langgraph-interrupt-re-runs-the-node-on-resume)
- "Structured output calls fail trying to parse response content" — OpenAI Developer Community, 2025-01-05. <https://community.openai.com/t/structured-output-calls-fail-trying-to-parse-response-content/1080922> — cited in [Ch 2](../ch02/README.md#references)
- "The tools feature hallucinate more than the deprecated function_call" — OpenAI Community, 2023-11-07. <https://community.openai.com/t/the-tools-feature-hallucinate-more-than-the-deprecated-function-call-gpt-3-5/479911> — cited in [Ch 22 › D1](../ch22/README.md#d1-tool-argument-hallucination)
- "Time zones in the Semantic Layer" — dbt Community Forum, 2023-10-17 (maintainer reply 2023-11-04). <https://discourse.getdbt.com/t/time-zones-in-the-semantic-layer/10622> — cited in [Ch 7](../ch07/README.md#references)
- "Tips for handling finish_reason: length with JSON" — OpenAI Community, 2024-06-07. <https://community.openai.com/t/tips-for-handling-finish-reason-length-with-json/806445> — cited in [Ch 22 › A1](../ch22/README.md#a1-finish_reason--length-truncates-json)
- "Why is GPT-4 giving different answers with same prompt & temperature=0?" — OpenAI Community, 2023-04-06. <https://community.openai.com/t/why-is-gpt-4-giving-different-answers-with-same-prompt-temperature-0/143513> — cited in [Ch 22 › B2](../ch22/README.md#b2-nondeterminism-at-temperature-0)
- "Your AI does not know what 'today' means." — DEV Community, 2026-07-30. <https://dev.to/mads_hansen_27b33ebfee4c9/your-ai-does-not-know-what-today-means-neg> — cited in [Ch 22 › G1](../ch22/README.md#g1-timezone-today)

<!--
Maintainers: this appendix is built from the `## References` sections of ch01–ch22 and appendix-A.
When a chapter changes its References, re-merge it here (dedupe by URL) and update the counts in the
intro and the Contents table.

Same source cited under different URLs (old URL redirects; merged above):
- Anthropic "Prompt caching": ch17 uses docs.anthropic.com/en/docs/build-with-claude/prompt-caching, ch15 and ch22 use platform.claude.com/docs/en/build-with-claude/prompt-caching (redirects → platform.claude.com).
Consider updating ch17 to the canonical URL.
-->
