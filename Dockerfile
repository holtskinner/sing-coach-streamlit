# Use the official uv image with Python 3.14 preinstalled
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

EXPOSE 8080
WORKDIR /app

# uv config: compile bytecode for faster startup, copy (not link) from cache,
# and use the system-wide environment created in /app/.venv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# Install dependencies first (without the project) so this layer is cached
# and only re-runs when pyproject.toml or uv.lock change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy the application source and install the project itself
COPY . ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Put the virtualenv on PATH so we can invoke streamlit directly
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
