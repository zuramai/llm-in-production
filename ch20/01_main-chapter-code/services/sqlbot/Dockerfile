# ShopFlow data bot (/ask). Same multi-stage pattern as services/chat/Dockerfile.
#   docker build -t shopflow-sqlbot:1.0.0 ch20/01_main-chapter-code/services/sqlbot
ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 UV_NO_DEV=1
WORKDIR /app
# (With BuildKit/buildx you can swap --no-cache for a cache mount:
#  RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked)
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-cache

FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY ch20_sqlbot ./ch20_sqlbot
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER 10001
EXPOSE 8080
CMD ["uvicorn", "--factory", "ch20_sqlbot.app:create_app", "--host", "0.0.0.0", "--port", "8080", \
     "--timeout-graceful-shutdown", "25"]
