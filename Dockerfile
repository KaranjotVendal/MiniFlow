# Base image: CUDA 13.0 + cuDNN
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# Install system dependencies (curl is needed for the multi-stage copy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates \
    ffmpeg \
    git \
    libportaudio2 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Use a multi-stage build to copy the uv binary
# This is more efficient and secure than downloading it during the build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# Working directory
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy dependency metadata first for better layer caching.
COPY pyproject.toml uv.lock ./

# Install only dependencies first. This layer is reused unless lockfile changes.
RUN uv sync --frozen --no-install-project

# Copy project source after dependencies.
COPY . /app

# Install the project itself into the existing virtualenv.
RUN uv sync --frozen

# Create and use non-root user for runtime security.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# Container healthcheck using FastAPI liveness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Use `uv run` to start the application. It will automatically find and
# use the project's virtual environment.
# FastAPI entrypoint
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
