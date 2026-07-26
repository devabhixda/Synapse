FROM ghcr.io/astral-sh/uv:0.8.4 AS uv

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml ./
RUN uv sync --all-groups --no-install-project

COPY alembic.ini ./
COPY migrations ./migrations
COPY src ./src
COPY tests ./tests
RUN uv sync --all-groups

RUN useradd --create-home --uid 10001 synapse && chown -R synapse:synapse /app
USER synapse

CMD ["uvicorn", "synapse.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
