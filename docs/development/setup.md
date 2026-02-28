# Development Setup

This guide helps you set up the MiniFlow development environment.

## Prerequisites

- Python 3.13+
- Git
- CUDA-capable GPU (optional, for full pipeline testing)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/KaranjotVendal/MiniFlow.git
cd MiniFlow
```

### 2. Install Dependencies

We use [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Or with dev dependencies (includes ruff, pre-commit, pytest)
uv sync --group dev
```

### 3. Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Verify installation
ls .git/hooks/pre-commit
```

### 4. Verify Setup

```bash
# Run tests
uv run pytest tests/unit_tests -v

# Check linting
uv run ruff check src tests

# Run full validation
uv run python scripts/validate_pr.py --skip-e2e
```

## Development Workflow

### Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes**

3. **Run checks locally**
   ```bash
   # Format code
   uv run ruff format src tests

   # Check linting
   uv run ruff check src tests

   # Run tests
   uv run pytest tests/unit_tests -v
   ```

4. **Commit** (pre-commit hooks run automatically)
   ```bash
   git add .
   git commit -m "feat: add my feature"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   ```

## Project Structure

```
MiniFlow/
├── src/                    # Main source code
│   ├── app.py             # FastAPI application
│   ├── sts_pipeline.py    # Speech-to-speech pipeline
│   ├── stt/               # Speech-to-text
│   ├── tts/               # Text-to-speech
│   ├── llm/               # LLM integration
│   └── benchmark/         # Benchmarking framework
├── tests/                 # Test suite
│   ├── unit_tests/        # Unit tests
│   └── integration_tests/ # Integration tests
├── configs/               # Configuration files
├── scripts/               # Utility scripts
├── docs/                  # Documentation
└── vibevoice/            # Git submodule (TTS)
```

## Common Commands

### Running the API

```bash
# Development mode with auto-reload
uv run uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

# Production mode
MINIFLOW_DEVICE=cuda uv run uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# All unit tests
uv run pytest tests/unit_tests -v

# Specific test file
uv run pytest tests/unit_tests/stt/test_stt_pipeline.py -v

# With coverage
uv run pytest tests/unit_tests --cov=src
```

### Docker

```bash
# Build image
docker build -t miniflow:latest .

# Run container
docker run -p 8000:8000 --gpus all miniflow:latest

# Or use docker-compose
docker-compose up -d api
```

### Code Quality

```bash
# Format code
uv run ruff format src tests

# Check linting
uv run ruff check src tests

# Fix auto-fixable issues
uv run ruff check --fix src tests

# Run pre-commit manually
uv run pre-commit run --all-files
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIFLOW_DEVICE` | `cuda` | Device for inference (`cuda` or `cpu`) |
| `MINIFLOW_CONFIG` | `configs/baseline.yml` | Path to config file |
| `MINIFLOW_REQUEST_TIMEOUT_SECONDS` | `600` | Request timeout |
| `RELEASE_ID` | `dev` | Release identifier |
| `COQUI_TOS_AGREED` | - | Agree to Coqui TTS terms |

## IDE Setup

### VS Code

Recommended extensions:
- Ruff (for linting and formatting)
- Python
- Pylance
- YAML

Settings:
```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "charliermarsh.ruff",
  "ruff.organizeImports": true
}
```

### PyCharm

1. Install Ruff plugin
2. Configure external tools for formatting
3. Enable "Reformat code" on save

## Troubleshooting

### Import Errors

```bash
# Ensure you're in the project root
export PYTHONPATH=.

# Or use uv run
uv run python -c "import src.app"
```

### CUDA Issues

```bash
# Check CUDA availability
uv run python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
export MINIFLOW_DEVICE=cpu
```

### Pre-commit Not Running

```bash
# Reinstall hooks
uv run pre-commit install

# Or run manually
uv run pre-commit run --all-files
```

## Additional Documentation

- [Linting and Formatting](linting-formatting.md)
- [Pre-commit Hooks](pre-commit-hooks.md)
- [Architecture](../ARCHITECTURE.md)
- [Benchmarking](../benchmarking.md)

## Getting Help

- Open an issue on GitHub
- Check existing documentation
- Review similar PRs for patterns
