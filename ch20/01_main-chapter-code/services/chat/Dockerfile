# ShopFlow support chatbot. Multi-stage: uv resolves the locked dependencies in a builder stage;
# the runtime stage gets only the virtualenv, the code and a non-root user.
#   docker build -t shopflow-chat:1.0.0 ch20/01_main-chapter-code/services/chat
ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 UV_NO_DEV=1
WORKDIR /app
# Dependencies only: this layer stays cached until pyproject.toml or uv.lock change.
# (With BuildKit/buildx you can swap --no-cache for a cache mount:
#  RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked)
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
EXPOSE 8080
CMD ["uvicorn", "--factory", "ch20_chat.app:create_app", "--host", "0.0.0.0", "--port", "8080", \
     "--timeout-graceful-shutdown", "25"]
