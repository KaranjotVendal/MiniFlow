# Docker Guide for ML Projects

A comprehensive guide to Docker fundamentals and best practices for machine learning deployments.

## Table of Contents

1. [Docker Fundamentals](#docker-fundamentals)
2. [Core Concepts](#core-concepts)
3. [ML-Specific Best Practices](#ml-specific-best-practices)
4. [Security Considerations](#security-considerations)
5. [Common Patterns](#common-patterns)
6. [Docker Compose](#docker-compose)
7. [Quick Reference](#quick-reference)

---

## Docker Fundamentals

### What is Docker?

Docker packages applications with all dependencies into "containers" - lightweight, isolated environments that run consistently across different machines. Unlike virtual machines, containers share the host OS kernel, making them much more efficient.

### Why Use Docker for ML?

- **Reproducibility**: Same environment from development to production
- **Dependency Management**: Isolate Python/CUDA/cuDNN versions per project
- **Scalability**: Easy deployment across multiple machines
- **Collaboration**: Share exact environments with teammates
- **GPU Support**: Native NVIDIA GPU access in containers

---

## Core Concepts

| Term | Definition | Analogy |
|------|------------|---------|
| **Image** | Read-only template with app + dependencies | Class in OOP |
| **Container** | Running instance of an image | Object in OOP |
| **Dockerfile** | Instructions to build an image | Recipe |
| **Layer** | Each command creates a cached layer | Step in recipe |
| **Volume** | Persistent storage outside container | External hard drive |
| **Registry** | Repository for sharing images | GitHub for images |

### Basic Commands

```bash
# Build image from Dockerfile
docker build -t myapp .

# Run container with port mapping
docker run -p 8000:8000 myapp

# Run with GPU support
docker run --gpus all myapp

# List running containers
docker ps

# View container logs
docker logs <container_id>

# Enter running container
docker exec -it <id> bash

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>

# Remove image
docker rmi myapp
```

---

## ML-Specific Best Practices

### 1. Base Image Selection

Choose base images based on your needs:

```dockerfile
# GPU-enabled with CUDA/cuDNN
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Pre-installed PyTorch
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# CPU-only, smaller footprint
FROM python:3.11-slim

# TensorFlow optimized
FROM tensorflow/tensorflow:2.15.0-gpu
```

**Why this matters**: Official ML images include optimized CUDA/cuDNN libraries, saving hours of manual setup and avoiding version conflicts.

### 2. Multi-Stage Builds

Separate build dependencies from runtime:

```dockerfile
# Stage 1: Build
FROM python:3.11 as builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime (smaller)
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY src/ /app/src/
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "-m", "src.app"]
```

**Benefits**:
- Smaller final images (no build tools like gcc)
- Faster deployments
- Reduced attack surface

### 3. Layer Caching Strategy

Order Dockerfile commands by change frequency:

```dockerfile
# GOOD - Leverages cache effectively
COPY requirements.txt .                    # Rarely changes
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/                        # Changes frequently

# BAD - Invalidates cache on any code change
COPY . /app
RUN pip install -r requirements.txt
```

**How caching works**: Docker caches each layer. If a layer and all previous layers haven't changed, Docker reuses the cached version.

### 4. Model and Data Management

**Don't** include large models in images:

```dockerfile
# BAD - Bloated image
COPY models/10GB-model.pt /app/models/
```

**Do** use external storage:

```dockerfile
# Dockerfile
ENV MODEL_PATH=/models
VOLUME ["/models", "/data"]

# Runtime
docker run -v /host/models:/models -v /host/data:/data myapp
```

**Options**:
- **Volumes** (`-v`): Persistent container storage
- **Bind mounts**: Direct host filesystem access
- **Cloud storage**: Download at runtime (S3, GCS, HuggingFace Hub)

### 5. Python Environment Variables

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1    # Don't write .pyc files
ENV PYTHONUNBUFFERED=1           # Unbuffered output
ENV PYTHONHASHSEED=0             # Reproducible hashing
ENV PIP_NO_CACHE_DIR=1           # Don't cache pip packages
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
```

**Why**:
- `.pyc` files waste space in ephemeral containers
- Unbuffered output ensures real-time logs in `docker logs`
- Reproducible hashing helps with debugging

### 6. GPU Support Configuration

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

Run with GPU:
```bash
# All GPUs
docker run --gpus all myapp

# Specific GPUs
docker run --gpus '"device=0,1"' myapp

# With memory limits
docker run --gpus all --memory=16g --cpus=4 myapp
```

---

## Security Considerations

### Why Non-Root Users Matter

Running as root in containers is dangerous because:

1. **Container Breakout**: Root inside container = potential root on host
2. **Privilege Escalation**: Kernel exploits can escape container boundaries
3. **Supply Chain Attacks**: Malicious packages run with full privileges
4. **Defense in Depth**: Principle of least privilege

### Implementing Non-Root Users

```dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Create app directory with proper permissions
WORKDIR /app
RUN chown appuser:appgroup /app

# Switch to non-root user
USER appuser

# Install dependencies as user
COPY --chown=appuser:appgroup requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appgroup . .

# Add local bin to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

CMD ["python", "app.py"]
```

### Additional Security Hardening

```bash
# Run with security options
docker run \
  --read-only \                          # Read-only filesystem
  --cap-drop=ALL \                       # Drop all capabilities
  --security-opt=no-new-privileges:true \ # Prevent privilege escalation
  --user 1000:1000 \                     # Explicit user ID
  myapp
```

### Secrets Management

**Don't** hardcode secrets:

```dockerfile
# BAD
ENV API_KEY=sk-12345
```

**Do** use environment variables or secrets:

```bash
# Runtime injection
docker run -e API_KEY=$API_KEY myapp

# Or use Docker secrets (Swarm/Kubernetes)
docker run --secret api_key myapp
```

---

## Common Patterns

### Pattern 1: FastAPI Model Serving

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser
WORKDIR /app

# Install dependencies
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy application code
COPY --chown=appuser:appuser src/ ./src/

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Pattern 2: Training Job

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install training dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY train.py .
COPY config/ ./config/

# Create directories for data and outputs
RUN mkdir -p /data /outputs

# Entrypoint for training
ENTRYPOINT ["python", "train.py"]
CMD ["--config", "config/default.yaml"]
```

Run with:
```bash
docker run --gpus all \
    -v $(pwd)/data:/data \
    -v $(pwd)/outputs:/outputs \
    my-train-image \
    --data-dir /data --output-dir /outputs
```

### Pattern 3: Development Environment

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/workspace
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ~/.cache/torch:/root/.cache/torch
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=all
    ports:
      - "8888:8888"  # Jupyter
      - "6006:6006"  # TensorBoard
      - "8000:8000"  # API
    stdin_open: true
    tty: true
    command: bash
```

```dockerfile
# Dockerfile.dev
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install development tools
RUN pip install jupyterlab tensorboard matplotlib seaborn

# Keep container running for interactive use
CMD ["bash"]
```

---

## Docker Compose

Docker Compose orchestrates multi-container applications. While Docker packages individual containers, Compose manages how they work together.

### Why Use Compose?

| Without Compose | With Compose |
|----------------|--------------|
| Complex `docker run` commands with many flags | Single `docker-compose up` command |
| Manual networking setup | Automatic service discovery |
| Hard to share configurations | Version-controlled YAML files |
| Managing containers individually | Manage entire stack as one unit |

### Basic Structure

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgres://db:5432
    depends_on:
      - db

  db:
    image: postgres:15
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

### Core Concepts

| Concept | Purpose | Example |
|---------|---------|---------|
| **Service** | Container definition | `api`, `db`, `redis` |
| **Image** | What to run | `postgres:15` or `build: .` |
| **Ports** | Expose container ports | `8000:8000` |
| **Volumes** | Persistent storage | `./data:/data` |
| **Environment** | Configuration | `DATABASE_URL=...` |
| **Networks** | Container communication | Auto-created between services |
| **Depends_on** | Startup order | `api` starts after `db` |

### ML-Specific Compose Features

**GPU Support:**
```yaml
services:
  training:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Model Caching:**
```yaml
volumes:
  huggingface-cache:

services:
  api:
    volumes:
      - huggingface-cache:/root/.cache/huggingface
```

**Development vs Production:**
```bash
# Production
docker-compose up -d

# Development with hot reload
docker-compose -f docker-compose.dev.yml up
```

### Common Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Restart specific service
docker-compose restart api

# Stop everything
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## Quick Reference

### Dockerfile Instructions

| Instruction | Purpose | Example |
|-------------|---------|---------|
| `FROM` | Base image | `FROM python:3.11` |
| `RUN` | Execute command | `RUN pip install torch` |
| `COPY` | Copy files from host | `COPY . /app` |
| `ADD` | Copy + extract URLs/archives | `ADD https://... /data` |
| `WORKDIR` | Set working directory | `WORKDIR /app` |
| `ENV` | Set environment variable | `ENV PYTHONUNBUFFERED=1` |
| `EXPOSE` | Document port usage | `EXPOSE 8000` |
| `USER` | Set user for subsequent commands | `USER appuser` |
| `VOLUME` | Create mount point | `VOLUME ["/data"]` |
| `CMD` | Default command | `CMD ["python", "app.py"]` |
| `ENTRYPOINT` | Fixed command prefix | `ENTRYPOINT ["python"]` |
| `HEALTHCHECK` | Container health monitoring | `HEALTHCHECK CMD curl...` |

### Common docker run Flags

```bash
# Port mapping
-p host_port:container_port

# Volume mounting
-v host_path:container_path

# Environment variables
-e VAR=value

# GPU access
--gpus all

# Resource limits
--memory=16g --cpus=4

# Detached mode
-d

# Interactive mode
-it

# Remove container after exit
--rm

# Container name
--name mycontainer

# Network
--network host
```

### .dockerignore

The `.dockerignore` file tells Docker which files to exclude from the build context. This is critical for ML projects because:

1. **Speed**: Smaller context = faster builds
2. **Size**: Don't bloat images with unnecessary files
3. **Security**: Prevent secrets from being included
4. **Cache**: Irrelevant changes won't invalidate Docker cache

**See the full reference**: [`.dockerignore` Reference](./DOCKERIGNORE.md)

**Quick example**:
```gitignore
# Never include in Docker context
.git
__pycache__
*.pyc
.venv
.env
*.pt
*.pth
data/
models/
```

### Docker Compose

Docker Compose orchestrates multi-container applications. While Docker packages individual containers, Compose manages how they work together.

**Key benefits:**
- Single command to start entire stack
- Automatic service discovery and networking
- Version-controlled configuration
- Separate dev/prod environments

**See the full guide**: [Docker Compose Guide](./COMPOSE.md)

**Quick example**:
```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgres://db:5432
    depends_on:
      - db

  db:
    image: postgres:15
```

---

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Python on Docker Hub](https://hub.docker.com/_/python)
- [NVIDIA CUDA Images](https://hub.docker.com/r/nvidia/cuda)

---

## Troubleshooting

### Out of Memory

```bash
# Check container memory usage
docker stats

# Run with memory limit
docker run --memory=8g --memory-swap=8g myapp
```

### GPU Not Available

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check Docker daemon configuration
# /etc/docker/daemon.json should include:
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

### Permission Denied

```bash
# Fix ownership in container
RUN chown -R appuser:appuser /app

# Or run with user ID mapping
docker run --user $(id -u):$(id -g) myapp
```

### Large Image Size

```bash
# Check layer sizes
docker history myimage

# Use multi-stage builds
# Use .dockerignore
# Use slim base images
# Clean up in same RUN layer:
RUN apt-get update && apt-get install -y package \
    && rm -rf /var/lib/apt/lists/*
```
