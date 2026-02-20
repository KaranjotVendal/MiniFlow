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

# Copy the project definition and the lockfile
COPY pyproject.toml uv.lock ./

# Copy the rest of your application code
COPY . /app

# Sync the project. This single command will:
# 1. Create a virtual environment at .venv
# 2. Install all dependencies from uv.lock
# 3. Install the current project, making 'app' importable
RUN uv sync --frozen

# Use `uv run` to start the application. It will automatically find and
# use the project's virtual environment.
# FastAPI entrypoint
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
