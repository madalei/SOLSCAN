# Build context is the repo root (docker-compose sets this); needs helpers/ as a sibling of app/.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /usr/local/bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Only what app/streamlit_app.py's import chain actually needs at runtime
COPY helpers/ helpers/
COPY app/ app/
RUN uv sync --frozen --no-dev

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
