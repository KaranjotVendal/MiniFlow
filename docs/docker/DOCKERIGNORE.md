# .dockerignore Reference

Complete guide to `.dockerignore` for ML projects.

## What is .dockerignore?

`.dockerignore` tells Docker which files to **exclude** from the build context. When you run `docker build`, Docker sends your entire project directory to the Docker daemon. This file prevents unnecessary files from being included.

## Why It Matters

| Benefit | Explanation |
|---------|-------------|
| **Faster builds** | Smaller context = faster transfer to daemon |
| **Smaller images** | Don't bloat with dev/test files |
| **Security** | Don't leak secrets or sensitive data |
| **Cache efficiency** | Unnecessary file changes won't invalidate cache |

## How It Works

```
Project Directory          Docker Build Context
├── src/                   ├── src/          (included)
├── models/                ├── models/       (excluded by .dockerignore)
│   └── 10GB-model.pt
├── data/
│   └── dataset.csv
├── .env
├── .git/
└── .dockerignore    ────→  (not sent to daemon)
```

## Pattern Syntax

```gitignore
# Comment - ignored by Docker
*.log           # All .log files
!important.log  # Except important.log
temp?           # temp1, temp2, etc.
**/*.pyc        # All .pyc in any directory
build/          # Directory named build
!important/     # Except directory named !important
```

## ML Project Template

```gitignore
# =============================================================================
# VERSION CONTROL
# =============================================================================
.git
.gitignore
.gitmodules
.gitattributes
.github/

# =============================================================================
# PYTHON ENVIRONMENT & CACHE
# =============================================================================
# Virtual environments (use container's Python instead)
.venv/
venv/
env/
ENV/
.python-version

# Python cache and compiled files
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Testing and linting caches
.pytest_cache/
.ruff_cache/
.mypy_cache/
.dmypy.json
.coverage
.coverage.*
htmlcov/
.tox/
.nox/

# =============================================================================
# IDE & EDITOR FILES
# =============================================================================
.vscode/
.idea/
*.swp
*.swo
*~
.zed/
*.sublime-*

# =============================================================================
# ML/DL SPECIFIC - MODELS & DATA
# =============================================================================
# Large model files (mount as volumes or download at runtime)
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
*.h5
*.pb
*.onnx
*.tflite
*.mlmodel

# Data directories (mount as volumes)
data/
data_assets/
datasets/
raw_data/
processed_data/

# Experiment outputs
experiments/
runs/
wandb/
mlruns/
outputs/
results/
checkpoints/
sweeps/

# Cache directories
.cache/
*.cache

# =============================================================================
# LOGS & TEMPORARY FILES
# =============================================================================
*.log
logs/
log/
temp_*
*.tmp
*.temp
*.wav
*.mp3
*.mp4

# =============================================================================
# DOCUMENTATION & DEVELOPMENT
# =============================================================================
# Documentation (not needed at runtime)
docs/
*.md
!README.md
!LICENSE

# Jupyter notebooks
*.ipynb
.ipynb_checkpoints/

# Development scripts
scripts/

# Local configuration
.env
.env.*
!.env.example
*.local
config.local.yaml

# =============================================================================
# TESTING
# =============================================================================
tests/
test/
*_test.py
*_tests.py
test_*.py

# =============================================================================
# BUILD ARTIFACTS
# =============================================================================
dist/
build/
*.egg-info/
*.egg
*.whl

# =============================================================================
# OS SPECIFIC
# =============================================================================
.DS_Store
Thumbs.db

# =============================================================================
# SECURITY - NEVER INCLUDE THESE
# =============================================================================
# Secrets and credentials
*.pem
*.key
*.crt
*.p12
*.pfx
secrets/
credentials/
.aws/
.gcp/
.kube/

# SSH keys
.ssh/
id_rsa
id_dsa
id_ecdsa
id_ed25519
```

## Category Explanations

### Version Control
- **Why exclude**: Git history can be 100MB+. Not needed at runtime.
- **What stays**: None

### Python Environment
- **Why exclude**: Containers use their own Python installation
- **What stays**: `pyproject.toml`, `requirements.txt`, `uv.lock`

### IDE Files
- **Why exclude**: Developer-specific, not needed in container
- **What stays**: None

### ML Models & Data
- **Why exclude**: Models can be 1GB-100GB+. Mount as volumes instead
- **What stays**: Model loading code, configuration

### Logs & Temp Files
- **Why exclude**: Generated at runtime, cause cache invalidation
- **What stays**: None

### Documentation
- **Why exclude**: Not needed for runtime, can be large
- **What stays**: `README.md`, `LICENSE` (small, useful)

### Testing
- **Why exclude**: Not needed in production images
- **What stays**: None (use multi-stage builds if tests needed)

### Security
- **Why exclude**: Prevent credential leaks in image layers
- **What stays**: None

## Common Mistakes

### 1. Forgetting .dockerignore
```bash
# Without .dockerignore: 5GB context
# With .dockerignore: 50MB context
# Result: 100x faster builds
```

### 2. Including .env files
```gitignore
# BAD - Will leak secrets
.env

# GOOD - Explicit exclusion
.env
.env.local
.env.production
```

### 3. Not excluding cache directories
```gitignore
# BAD - Cache invalidation on every build
__pycache__/

# GOOD - Proper exclusion
__pycache__/
*.pyc
.pytest_cache/
```

### 4. Excluding too much
```gitignore
# BAD - Will break the build
src/
config/

# GOOD - Only exclude what shouldn't be in image
__pycache__/
*.pyc
```

## Best Practices

1. **Start with a template** - Use the ML template above
2. **Review regularly** - Update as project evolves
3. **Test build context size**:
   ```bash
   docker build -t test . --progress=plain 2>&1 | head -20
   ```
4. **Use exceptions wisely**:
   ```gitignore
   *.md
   !README.md  # Keep README
   ```
5. **Document exclusions** - Add comments explaining why

## Verification

Check what's being sent to Docker:

```bash
# See build context
docker build -t test . --progress=plain

# Check image contents
docker run --rm test ls -la

# Compare sizes
du -sh .                    # Total project
docker build -t test .      # Build
docker images test          # Image size
```

## Project-Specific Examples

### FastAPI Service
```gitignore
# Exclude
tests/
docs/
*.ipynb
models/          # Mount instead
data/            # Mount instead

# Include
src/
config/
pyproject.toml
```

### Training Pipeline
```gitignore
# Exclude
outputs/         # Generated artifacts
wandb/           # Experiment tracking
checkpoints/     # Model checkpoints

# Include
src/
configs/
train.py
```

### Jupyter Development
```gitignore
# Exclude
.ipynb_checkpoints/
*.ipynb          # Keep notebooks in dev only

# Include
src/
notebooks/       # If sharing examples
```

## See Also

- [Docker Build Context](https://docs.docker.com/build/building/context/)
- [.dockerignore File](https://docs.docker.com/engine/reference/builder/#dockerignore-file)
- [Docker Security](https://docs.docker.com/engine/security/)
