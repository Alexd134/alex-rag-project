# FROM ghcr.io/astral-sh/uv:latest AS uv

# # FROM python:3.13-slim
# FROM public.ecr.aws/lambda/python:3.11 AS builder

# # COPY --from=uvbin /uv /uvx /bin/

# # WORKDIR /app
# WORKDIR ${LAMBDA_TASK_ROOT}

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PIP_NO_CACHE_DIR=1 \
#     IS_USING_IMAGE_RUNTIME=True \
#     CHROMA_PATH="data/chroma" \
#     UV_SYSTEM_PYTHON=1

# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     build-essential \
# #     ca-certificates \
# #     curl \
# #  && rm -rf /var/lib/apt/lists/*

# RUN --mount=from=uv,source=/uv,target=/bin/uv \
#     --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=uv.lock,target=/workspace/uv.lock \
#     --mount=type=bind,source=pyproject.toml,target=/workspace/pyproject.toml \
#     bash -lc 'cd /workspace \
#       && uv export --frozen --no-emit-workspace --no-dev --no-editable -o requirements.txt \
#       && uv pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"'



# # Stage 2: final runtime
# FROM public.ecr.aws/lambda/python:3.13

# COPY --from=builder ${LAMBDA_TASK_ROOT} ${LAMBDA_TASK_ROOT}

# COPY src/ ${LAMBDA_TASK_ROOT}/src

# ENV PYTHONPATH="${LAMBDA_TASK_ROOT}/src"

# CMD ["src/api_handler.handler"]


# # COPY pyproject.toml ./
# # COPY uv.lock ./
# # COPY README.md ./

# # # RUN uv sync --frozen

# # RUN uv pip install --frozen -t {$LAMBDA_TASK_ROOT} .

# # COPY src/ ./src/

# # # RUN python -m compileall -q /app

# # # RUN mkdir -p /app/data/chroma

# # EXPOSE 8000

# # # CMD ["uv", "run", "uvicorn", "api_handler:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]

# # CMD ["src/api_handler.handler"]

# docker-images/lambda.Dockerfile
# syntax=docker/dockerfile:1.7

# Stage 0: provide uv binary
FROM ghcr.io/astral-sh/uv:0.9.3 AS uvbin

# Stage 1: build deps into Lambda task root
FROM public.ecr.aws/lambda/python:3.13 AS builder

# Speed + determinism
ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_INSTALLER_METADATA=1 \
    UV_LINK_MODE=copy

# Work inside a scratch workspace (bind-mounted files land here)
WORKDIR /workspace

# Use BuildKit mounts:
# - mount uv binary from the uvbin stage
# - cache uv downloads
# - bind pyproject & lock from the build context
RUN --mount=from=uvbin,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    sh -lc 'uv export --frozen --no-emit-workspace --no-dev --no-editable -o requirements.txt && \
            uv pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"'

# Stage 2: final runtime
FROM public.ecr.aws/lambda/python:3.13

# Bring in site-packages installed to the task root
COPY --from=builder ${LAMBDA_TASK_ROOT} ${LAMBDA_TASK_ROOT}

# Copy your app (src/ layout)
COPY src/ ${LAMBDA_TASK_ROOT}/src

# Ensure src/ is importable and use Lambda-safe temp dir for Chroma
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}/src" \
    CHROMA_PATH="/tmp/chroma" \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

CMD ["src/api_handler.handler"]
