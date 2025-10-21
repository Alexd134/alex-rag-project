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
COPY src/data/chroma/ ${LAMBDA_TASK_ROOT}/src/data/chroma/


# Ensure src/ is importable
# Note: CHROMA_PATH and IS_USING_IMAGE_RUNTIME are set via CDK environment variables
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

CMD ["src/api_handler.handler"]
