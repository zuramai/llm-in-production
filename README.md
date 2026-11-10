# LLM in Production

**A free, hands-on book about shipping large language models in real systems.**

[![Download the PDF](https://img.shields.io/badge/Download-PDF-d63031?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://github.com/zuramai/llm-in-production/releases/latest/download/llm-in-production.pdf)
[![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue?style=for-the-badge)](LICENSE)
[![Prose: CC BY-SA 4.0](https://img.shields.io/badge/prose-CC%20BY--SA%204.0-lightgrey?style=for-the-badge)](LICENSE-CONTENT)

22 chapters, each built, run and measured. Each one takes a problem you'll meet when you put an LLM
behind a real product, solves it end to end on a realistic online shop, and shows the numbers:
accuracy, latency and cost, measured, not guessed. Every decision explains the alternative it
rejected.

**[📄 Download the PDF](https://github.com/zuramai/llm-in-production/releases/latest/download/llm-in-production.pdf)** · read it on GitHub below · run every notebook yourself

## What you'll build

| | Chapter | You build |
|---|---|---|
| **I · Foundations** | [1. From API call to production chatbot](ch01/README.md) | a streaming FAQ support bot with memory, timeouts and fallbacks |
| | [2. Structured outputs](ch02/README.md) | ticket triage into validated JSON, four methods compared |
| | [3. Evaluating LLM systems](ch03/README.md) | a regression gate and a calibrated LLM judge |
| **II · Data & retrieval** | [4. RAG over internal docs](ch04/README.md) | hybrid search with reranking and citations |
| | [5. Agentic RAG](ch05/README.md) | a LangGraph agent that combines documents and SQL |
| | [6. Text-to-SQL Slack bot](ch06/README.md) | a bot that answers "how many orders today?" safely |
| | [7. Semantic layers](ch07/README.md) | governed metrics the model can't get wrong |
| | [8. Documents & vision](ch08/README.md) | invoice extraction from PDFs and scans |
| **III · Agents & tools** | [9. Tool calling & MCP](ch09/README.md) | a support assistant with approval-gated refunds |
| | [10. Reliable agents](ch10/README.md) | a checkpointed agent with human-in-the-loop |
| | [11. Incident analysis agent](ch11/README.md) | an on-call agent that reads Prometheus and Loki |
| | [12. Code-review bot](ch12/README.md) | a pull-request reviewer that finds planted bugs line by line |
| **IV · Hardening** | [13. Guardrails & security](ch13/README.md) | 40 attacks, 5 defenses, measured |
| | [14. Observability & LLMOps](ch14/README.md) | tracing, feedback and prompt experiments in Langfuse |
| | [15. Gateway, caching & cost](ch15/README.md) | fallbacks, budgets, caches and a cheap-to-strong cascade |
| **V · Models & compute** | [16. Self-hosting open models](ch16/README.md) | Ollama vs llama.cpp vs vLLM vs SGLang on one GPU |
| | [17. GPU optimization](ch17/README.md) | quantization, prefix caching and speculative decoding |
| | [18. Fine-tuning your own model](ch18/README.md) | LoRA/QLoRA vs a well-prompted hosted model |
| | [19. Managed fine-tuning](ch19/README.md) | one dataset for OpenAI, Bedrock, Vertex and SageMaker |
| **VI · Scale** | [20. Deploying to production](ch20/README.md) | containers, Kubernetes, eval gates and canaries |
| | [21. Scaling to a million users](ch21/README.md) | load tests, rate limits, failover and job queues |
| | [22. Edge cases from the field](ch22/README.md) | 34 real failure modes, each with a repro and a fix |
| **Appendices** | [A](appendix-A/README.md) · [B](appendix-B/README.md) · [C](appendix-C/README.md) | model & price snapshot · glossary · references |

Every chapter runs against **ShopFlow**, a fictional online shop with a real Postgres database
(30,000 orders, payments, refunds, support tickets and docs) that one command seeds.

## Run it

```bash
uv sync                                     # Python 3.12 + all chapter dependencies
cp .env.example .env                        # add an OpenRouter key, or point it at local Ollama
docker compose -f datasets/shopflow/docker-compose.yml up -d
uv run python datasets/shopflow/seed.py --reset
uv run jupyter lab                          # open ch01/01_main-chapter-code/ch01.ipynb
```

You need Python, basic SQL and Docker. No ML background. Chapters 16–19 need an NVIDIA GPU.
Providers, tests, hardware and troubleshooting: **[setup guide](setup/README.md)**.

## Credits and license

Inspired by Sebastian Raschka's *Build a Large Language Model (From Scratch)* and Chip Huyen's
*AI Engineering*. Code under [Apache-2.0](LICENSE); prose and images under
[CC BY-SA 4.0](LICENSE-CONTENT).
