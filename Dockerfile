FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# OpenMP runtime, needed by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps before copying code so this layer stays cached
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY . .
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "chrono24"]
