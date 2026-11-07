# Style guide: the contract for every chapter

Every chapter of *LLM in Production* follows this guide. If something here conflicts with your
instinct, follow this guide. If something here is wrong, change this guide first, then the
chapters.

## Voice

- A **senior AI solution engineer talking to a backend engineer** over a whiteboard. The reader
  knows Python, HTTP, SQL and Docker. They don't know transformers, sampling or evals yet.
- **Plain language first.** Define every term on first use, in one sentence plus an analogy.
  For example: "The *context window* is the model's working memory: everything it can see at
  once, like the RAM of a single request."
- Use **"we"**: "We add a guard", "we measure". The reader builds alongside us.
- Show the **thought process**: what we tried, what broke, what the numbers said, what we chose
  and why. A chapter that only shows the final answer has failed.
- Short paragraphs. One idea per paragraph. Prefer concrete numbers to adjectives.
- No marketing ("blazing fast", "game-changing"), no filler ("In this section we will...").

## Chapter skeleton

```
# Chapter N: Title
> **Tested on:** ...                         (badge, see below)

Opening hook: a real scenario + what is happening in the industry right now (dated, sourced).

## The problem
## Requirements                              functional / non-functional table with numbers
## The data                                  schema (Mermaid ER), seed commands, sample rows
## Concepts                                  the theory needed, in plain language, with visuals
## v0: the naive solution
## Measuring v0                              real failures from real runs
## Iterating                                 one ### per improvement, each with a decision box
## Production hardening
## Serving it
## Results                                   eval table, real numbers
## Exercises                                 2-4
## What we learned                           5-10 bullets
## References
```

Each `### Iteration` subsection follows the same pattern: what is wrong → options → decision
box → code excerpt → measured effect.

## Callouts

Use GitHub alert syntax so the callouts render on github.com:

```markdown
> [!IMPORTANT]
> **Decision: SSE over WebSockets** — one-way stream, plain HTTP, works through every proxy.
> WebSockets would give us bidirectional messages we don't need.

> [!NOTE]
> **In the wild:** In February 2024 a Canadian tribunal held Air Canada liable for a refund
> policy its chatbot invented ([source](https://...)).

> [!WARNING]
> Reasoning models reject `temperature`. ...

> [!TIP]
> Set `max_retries=0` in tests so failures surface immediately.

> [!CAUTION]
> **Not executed by the author** — ...
```

- **Decision boxes** (`[!IMPORTANT]`): always `**Decision: X over Y** — why`. Name the
  alternative you rejected and the condition under which you'd pick it instead.
- **In the wild** (`[!NOTE]`): a real company, incident, paper or postmortem, with a link. If
  you can't source it, don't include it.
- At least **3 In-the-wild notes** and **3 decision boxes** per chapter.

## Badges

Executed chapters start with:

```markdown
> **Tested on:** 2026-11-07 · macOS (Apple M2 Pro, 16 GB) · Python 3.12 ·
> openai/gpt-6-luna, anthropic/claude-sonnet-5.5 via OpenRouter; qwen3:8b via Ollama 0.15 ·
> openai 3.46, fastapi 0.162
```

Code that was **not** executed (needs CUDA, a cloud account, a third-party workspace) carries
this badge directly above it:

```markdown
> [!CAUTION]
> **Not executed by the author** — requires an NVIDIA GPU with CUDA 12+. Written against
> <official doc URL> (accessed 2026-11-07) and checked with `python -m py_compile`.
```

## Visuals

Every architecture, data flow, lifecycle and comparison gets a visual. The reader should be able
to skim a chapter by its diagrams.

| What | How | Where |
|---|---|---|
| flows, sequences, state machines, ER diagrams | Mermaid (GitHub renders it) | inline in README |
| architecture, layouts, "layers" pictures | hand-written SVG | `chNN/images/NN-<slug>.svg` |
| charts of measured results | matplotlib, saved by the notebook | `chNN/images/NN-<slug>.png` |
| UI of apps we build | `tools/screenshot.py` (Playwright) | `chNN/images/NN-<slug>.png` |

SVG rules: set `viewBox`; use fixed colors that read on both light and dark GitHub themes
(mid-saturation fills with dark text inside boxes, lines in `#888`); always draw an opaque
background rect (`#ffffff` with rounded corners) so the diagram is legible on dark mode; no
external references, fonts or scripts; `font-family="Helvetica, Arial, sans-serif"`.

Never use third-party or copyrighted images. Never fabricate a screenshot. If the author
couldn't capture it, the chapter shows text output instead.

## Code in prose

- Code blocks in the README are **excerpts copied from tested files**. Name the file in a
  comment on the first line: `# ch06/01_main-chapter-code/sqlbot/guard.py`.
- **Outputs are pasted from real runs** (the notebook's saved outputs or a terminal run).
  Never write an output by hand. Trim long outputs with `...`.
- Keep excerpts short (≤ 40 lines). Link to the full file.
- Commands always start from the repo root unless they say `cd` first.

## Numbers, models, prices

- Every metric in the prose comes from a run recorded in the notebook or in `eval/results/`.
- Model names and prices carry an **"as of YYYY-MM"** stamp: "`openai/gpt-6-luna` ($0.10 / $0.50
  per 1M input/output tokens on OpenRouter, as of 2026-11)".
- Round sensibly: latency to ms, cost to 4 significant figures, accuracy to 1 decimal.

## Sources and research

- Before writing a chapter, research (a) current tool versions and APIs from official docs,
  (b) at least 3 verifiable in-the-wild facts, and (c) at least 3 practitioner pain points
  from Reddit, Hacker News, Stack Overflow or GitHub issues. Each one needs a URL.
- Quote at most 2 sentences from any source. Paraphrase and link instead.
- Every source used goes into `## References` as `- Title — Publisher, date. URL`.
- The two reference books that inspired this one are credited in the root README. They are
  never quoted at length.

## Code conventions

- Python 3.12, `ruff` at line length 100, type hints on public functions.
- Chat Completions (`client.chat.completions.create`) through `llmprod.chat_client()`, so
  every example works on OpenRouter, OpenAI and Ollama.
- Each chapter is self-contained. Code reused from earlier chapters is copied into
  `chNN/01_main-chapter-code/previous_chapters.py`, so chapters can be read in any order.
- Tests that call a real model are marked `@pytest.mark.llm`. Tests that need Docker
  services are marked `@pytest.mark.docker`.
- Notebooks are committed **with outputs**, executed headless top to bottom, and contain
  no error outputs.
- Secrets only come from `.env`. Never print an API key.
