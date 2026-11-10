# Setup

The whole book runs from one Python environment managed by [uv](https://docs.astral.sh/uv/).
Commands start from the repository root.

## 1. Install uv and the dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # or: brew install uv
uv sync                                             # creates .venv with Python 3.12
```

`uv sync` installs every chapter's dependency group (`[tool.uv] default-groups = "all"`).
Packages that need CUDA or a cloud SDK are **not** in `pyproject.toml`. Those chapters keep a
local `requirements-cuda.txt` or `requirements-cloud.txt`.

## 2. Configure a model provider

```bash
cp .env.example .env
```

All code talks to an **OpenAI-compatible Chat Completions endpoint**, so switching providers
only means editing `.env`. Pick one of the following.

### OpenRouter (default)

One key gives you hundreds of models (OpenAI, Anthropic, Google, Qwen, ...). Create a key at
<https://openrouter.ai/keys> and add credits.

```ini
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-...
LLM_MODEL=openai/gpt-6-luna
LLM_MODEL_STRONG=anthropic/claude-sonnet-5.5
```

### OpenAI direct

```ini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-6-luna
LLM_MODEL_STRONG=gpt-6.1-sol
```

(Model IDs as of 2026-11, from <https://developers.openai.com/api/docs/models>.)

### Ollama (local, free, offline)

```bash
curl -fsSL https://ollama.com/install.sh | sh   # Linux; macOS/Windows: download from ollama.com
ollama serve                                    # skip if the installer started the service
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Ollama uses an NVIDIA GPU automatically when the driver is installed (`nvidia-smi` works);
`ollama ps` shows `100% GPU` for a loaded model. It also runs on Apple Silicon and on CPU,
only slower. The book's local numbers come from one RTX 3090 (24 GB).

```ini
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen3:8b
LLM_MODEL_STRONG=qwen3:8b
```

Embeddings always come from Ollama's `nomic-embed-text` (`EMBED_*` variables), so you need
Ollama running for the retrieval chapters even if you use OpenRouter for chat.

Check the configuration:

```bash
uv run python -c "from llmprod import get_settings, chat_client; s=get_settings(); \
print(chat_client().chat.completions.create(model=s.llm_model, \
messages=[{'role':'user','content':'Say OK'}]).choices[0].message.content)"
```

## 3. Start the ShopFlow database

```bash
docker compose -f datasets/shopflow/docker-compose.yml up -d
uv run python datasets/shopflow/seed.py --reset
```

The seeder prints the anchor time and confirms how many orders were placed "today". "Today" is
the day you ran it, so run it again on a new day. See
[datasets/shopflow/README.md](../datasets/shopflow/README.md).

## 4. Notebooks

```bash
uv run jupyter lab
```

Open a chapter notebook and choose the **`.venv`** (Python 3.12) kernel. In VS Code, select
the interpreter at `.venv/bin/python`. Headless, saving outputs in place:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace ch01/01_main-chapter-code/ch01.ipynb
```

**Hardware.** Parts I–IV run on any laptop with Docker; model calls go to a hosted API
(OpenRouter by default) or to a local Ollama. Part V (chapters 16–19) needs an NVIDIA GPU; it was
tested on one RTX 3090 (24 GB) under Ubuntu 22.04. Those chapters keep their CUDA-only packages
in a chapter-local `requirements-cuda.txt`. Cloud code (SageMaker, Vertex) is marked **Not
executed by the author** wherever that applies.

## 5. Tests

```bash
uv run pytest                              # unit tests; never calls a model, never re-seeds the DB
uv run pytest -m "docker and not reseed"   # tests that need the ShopFlow Postgres container
uv run pytest -m llm                       # tests that call the configured model (costs money)
uv run pytest -m reseed datasets           # seed invariants: re-seed the DB, drop chapter tables
```

## 6. Build the PDF

```bash
uv run playwright install chromium         # once
uv run python tools/build_pdf.py           # writes dist/llm-in-production.pdf
```

## Repository layout

```
chNN/
  README.md                  # the chapter
  01_main-chapter-code/      # notebook + scripts used in the chapter
  02_bonus_*/                # optional extras
  images/                    # SVG diagrams and PNG screenshots
appendix-{A,B,C}/            # model snapshot, glossary, references
datasets/shopflow/           # shared database, seeder, docs corpus
src/llmprod/                 # tiny shared helpers: settings, clients, DB connection
docs/                        # style guide and chapter template
tools/                       # screenshot and PDF build scripts
```

## 7. Screenshots (optional)

The chapters' UI screenshots were made with Playwright. To reproduce them:

```bash
uv run playwright install chromium
uv run python tools/screenshot.py http://localhost:8001 /tmp/chat.png
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Bind for 0.0.0.0:5433 failed: port is already allocated` | another Postgres uses 5433 | stop it, or change the port in `datasets/shopflow/docker-compose.yml` **and** `DATABASE_URL*` in `.env` |
| `Connection refused` on `localhost:11434` | Ollama isn't running | `ollama serve` or start the Ollama app |
| `openai.AuthenticationError: Error code: 401` | wrong/missing key, or key for another provider | check `LLM_API_KEY` and `LLM_BASE_URL` belong together |
| `RuntimeError: Missing LLM_API_KEY` | `.env` not found or empty value | run from inside the repo; `cp .env.example .env` |
| `400 ... Unsupported parameter: 'temperature'` | reasoning models fix their sampling | remove `temperature`, or use a non-reasoning model (see Chapter 1) |
| `model not found` on Ollama | model not pulled | `ollama pull <model>` |
| "today" counts are 0 | the seed was anchored on an earlier day | `uv run python datasets/shopflow/seed.py --reset` |
