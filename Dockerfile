FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first 
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and config
COPY src/ ./src/
COPY config.py ./

EXPOSE 8000

CMD ["bash", "-c", "./src/run.sh"]