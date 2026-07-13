FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# install deps before copying code so this layer stays cached
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY . .
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-c", "from chrono24 import load_watches; print(load_watches().shape)"]
