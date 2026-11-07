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
brew install ollama          # or download from https://ollama.com
ollama serve                 # or start the Ollama app
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

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
the interpreter at `.venv/bin/python`.

## 5. Screenshots (optional)

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
