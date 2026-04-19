FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first 
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and config .
COPY src/ ./src/
RUN chmod +x ./src/run.sh #so permission given to run.sh
COPY config.py ./

EXPOSE 8000

CMD ["./src/run.sh"]
