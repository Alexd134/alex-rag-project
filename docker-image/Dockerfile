FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    IS_USING_IMAGE_RUNTIME=True \
    CHROMA_PATH="data/chroma" \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
 && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./

RUN uv sync --frozen

COPY src/ ./src/

RUN python -m compileall -q /app

RUN mkdir -p /app/data/chroma

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api_handler:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
