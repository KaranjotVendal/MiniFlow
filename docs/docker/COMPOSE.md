# Docker Compose Guide

Complete guide to Docker Compose for ML projects.

## What is Docker Compose?

Docker Compose is a tool for defining and running multi-container Docker applications. While Docker packages individual containers, Compose orchestrates how they work together.

## The Problem Compose Solves

### Without Compose (Painful)

```bash
# Start database
docker run -d \
  --name db \
  --network myapp-network \
  -e POSTGRES_PASSWORD=secret \
  -v postgres-data:/var/lib/postgresql/data \
  postgres:15

# Start cache
docker run -d \
  --name redis \
  --network myapp-network \
  redis:7

# Start API (with all the flags)
docker run -d \
  --name api \
  --network myapp-network \
  -p 8000:8000 \
  -e DATABASE_URL=postgres://db:5432 \
  -e REDIS_URL=redis://redis:6379 \
  -v ./models:/models \
  --gpus all \
  myapp-api
```

### With Compose (Clean)

```yaml
services:
  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=secret
    volumes:
      - postgres-data:/var/lib/postgresql/data

  redis:
    image: redis:7

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgres://db:5432
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - db
      - redis

volumes:
  postgres-data:
```

```bash
docker-compose up -d
```

## Core Concepts

### Services

A service is a container definition:

```yaml
services:
  api:           # Service name (used for networking)
    build: .     # Build from Dockerfile

  db:            # Another service
    image: postgres:15  # Use existing image
```

### Networking

Compose automatically creates a network. Services can reach each other by name:

```yaml
services:
  api:
    environment:
      # Connect to db service by name
      - DATABASE_URL=postgres://db:5432

  db:
    image: postgres:15
```

### Volumes

Two types of volumes in Compose:

**Named volumes** (managed by Docker):
```yaml
volumes:
  postgres-data:   # Created and managed by Docker

services:
  db:
    volumes:
      - postgres-data:/var/lib/postgresql/data
```

**Bind mounts** (host filesystem):
```yaml
services:
  api:
    volumes:
      - ./models:/models        # Relative path
      - /absolute/path:/data    # Absolute path
```

### Environment Variables

```yaml
services:
  api:
    # Inline environment variables
    environment:
      - DEBUG=1
      - DATABASE_URL=postgres://db:5432

    # Or from .env file
    env_file:
      - .env

    # Or mixed
    environment:
      - LOG_LEVEL=${LOG_LEVEL:-info}  # With default
```

## Complete ML Example

```yaml
version: '3.8'

services:
  # FastAPI inference service
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - COQUI_TOS_AGREED=1
      - MODEL_PATH=/models
      - CACHE_DIR=/cache
      - LOG_LEVEL=info
    volumes:
      # Mount models (read-only)
      - ./models:/models:ro
      # Persist cache between restarts
      - huggingface-cache:/cache
      # Mount config
      - ./config:/app/config:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    depends_on:
      - redis

  # Redis for caching
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    restart: unless-stopped

  # Optional: Monitoring with Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

volumes:
  huggingface-cache:
  redis-data:
  prometheus-data:
```

## Development vs Production


---

### Production (`docker-compose.yml`)

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=info
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
```

### Development (`docker-compose.dev.yml`)

```yaml
services:
  api:
    volumes:
      # Live code reload
      - .:/app
      # Protect container's virtualenv
      - /app/.venv
    environment:
      - LOG_LEVEL=debug
      - DEBUG=1
    command: uv run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
    # Increase shared memory for PyTorch
    shm_size: '2gb'
```

### Usage

```bash
# Production
docker-compose up -d

# Development
docker-compose -f docker-compose.dev.yml up

# Override base with dev
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## ML-Specific Patterns

### Pattern 1: Training Pipeline

```yaml
services:
  data-preprocessing:
    build: .
    volumes:
      - ./raw-data:/data/in
      - processed-data:/data/out
    command: python preprocess.py --input /data/in --output /data/out

  training:
    build: .
    volumes:
      - processed-data:/data
      - ./outputs:/outputs
      - model-cache:/root/.cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python train.py --data /data --output /outputs
    depends_on:
      - data-preprocessing

volumes:
  processed-data:
  model-cache:
```

### Pattern 2: Model Serving with Load Balancing

```yaml
services:
  api-1:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  api-2:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api-1
      - api-2
```

### Pattern 3: Jupyter Development Environment

```yaml
services:
  jupyter:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8888:8888"
    volumes:
      - .:/workspace
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ~/.cache/torch:/root/.cache/torch
    environment:
      - JUPYTER_ENABLE_LAB=yes
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

## Common Commands

```bash
# Start services
docker-compose up -d

# Start specific service
docker-compose up -d api

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f api

# Scale service
docker-compose up -d --scale api=3

# Restart service
docker-compose restart api

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Build images
docker-compose build

# Build without cache
docker-compose build --no-cache

# Execute command in service
docker-compose exec api bash

# Run one-off command
docker-compose run --rm api python test.py

# Validate compose file
docker-compose config

# List running containers
docker-compose ps
```

## Best Practices

### 1. Use Environment Files

```yaml
# docker-compose.yml
services:
  api:
    env_file:
      - .env
```

```bash
# .env (gitignored)
DATABASE_URL=postgres://localhost:5432
API_KEY=secret
DEBUG=false
```

### 2. Health Checks

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 3. Resource Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          cpus: '2'
          memory: 8G
```

### 4. Restart Policies

```yaml
services:
  api:
    restart: unless-stopped  # Always restart unless manually stopped

  db:
    restart: always          # Always restart
```

### 5. Named Volumes for Data

```yaml
volumes:
  postgres-data:
    driver: local

services:
  db:
    volumes:
      - postgres-data:/var/lib/postgresql/data
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs service-name

# Check config
docker-compose config

# Validate YAML
docker-compose config --quiet
```

### GPU Not Available

```yaml
# Ensure runtime is specified
services:
  api:
    runtime: nvidia  # For older Docker Compose
    # OR use deploy.resources (v3+)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Port Already in Use

```bash
# Find what's using port 8000
lsof -i :8000

# Use different port in compose
ports:
  - "8001:8000"  # Host:Container
```

## See Also

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Compose File Reference](https://docs.docker.com/compose/compose-file/)
- [Docker Compose CLI](https://docs.docker.com/compose/reference/)

---

## MiniFlow Configuration

MiniFlow uses environment variables for configuration. See the [Architecture Guide](../ARCHITECTURE.md#configuration-system) for complete details.

### Required Variables

| Variable | Description |
|----------|-------------|
| `MINIFLOW_CONFIG` | Path to pipeline YAML config (e.g., `configs/3_TTS-to-vibevoice.yml`) |
| `RELEASE_ID` | Release identifier (e.g., `prod-pseudo`, `v1.0.0`) |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIFLOW_REQUEST_TIMEOUT_SECONDS` | 120 | Request timeout in seconds |
| `MINIFLOW_MAX_AUDIO_UPLOAD_BYTES` | 10485760 (10MB) | Max upload size in bytes |

### Example: Production

```yaml
services:
  api:
    environment:
      - COQUI_TOS_AGREED=1
      - MINIFLOW_CONFIG=configs/3_TTS-to-vibevoice.yml
      - RELEASE_ID=v1.0.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Example: Development

```yaml
services:
  api:
    environment:
      - LOG_LEVEL=debug
      - MINIFLOW_CONFIG=configs/baseline.yml
      - RELEASE_ID=dev-local
      - HF_HOME=/app/.cache/huggingface
    volumes:
      - hf_cache:/app/.cache/huggingface
```



---

## UV Python Installation in Docker

### The Problem

By default, UV installs Python to `~/.local/share/uv/python/` which is owned by root. When the app runs as a non-root user (e.g., `appuser`), it cannot access that directory, causing startup failures.

### Solution: Project-Local Python

Set a project-local Python installation directory:

```dockerfile
ENV UV_PYTHON_INSTALL_DIR=/app/.python
```

Now:
- Python is installed under `/app/.python/` (within the app directory)
- Accessible by any user
- No permission issues

### Alternative: Per-Project Local (Development)

For development, you can use a per-project local installation:

```bash
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
uv python install 3.13
uv sync
```

This creates `.python/` in your project directory. Add to `.gitignore`:
```
.python/
```

Reference: https://docs.astral.sh/uv/concepts/python/
