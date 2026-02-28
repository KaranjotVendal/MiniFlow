# Linting and Formatting

This document describes the linting and formatting tools used in MiniFlow, along with how to set them up and use them.

## Overview

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting. Ruff is an extremely fast Python linter and code formatter written in Rust, designed to replace multiple tools (Flake8, Black, isort, pyupgrade, etc.) with a single, unified interface.

## Why Ruff?

- **Speed**: 10-100x faster than alternatives
- **Unified**: One tool for linting and formatting
- **Compatible**: Drop-in replacement for Flake8, Black, and isort
- **Modern**: Native support for Python 3.13+ features

## Configuration

Ruff is configured in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py313"  # Python version to target
line-length = 100         # Maximum line length

[tool.ruff.lint]
select = [
    "F",   # Pyflakes - basic error detection
    "E",   # pycodestyle - PEP 8 errors
    "B",   # Bugbear - common bug patterns
    "I",   # isort - import sorting
    "C4",  # Comprehensions - simplify comprehensions
    "UP",  # Pyupgrade - modern Python syntax
]
ignore = [
    "E501",  # Line too long (handled by formatter)
]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### Rule Categories

| Code | Category | Description |
|------|----------|-------------|
| `F` | Pyflakes | Basic error detection (undefined names, unused imports) |
| `E` | pycodestyle | PEP 8 style errors |
| `B` | Bugbear | Common bug patterns (mutable defaults, unused loop vars) |
| `I` | isort | Import sorting and organization |
| `C4` | Comprehensions | Simplify list/dict/set comprehensions |
| `UP` | Pyupgrade | Modern Python syntax (Python 3.13+) |

## Usage

### Running the Linter

```bash
# Check all files
uv run ruff check src tests

# Check with auto-fix (fixes auto-fixable issues)
uv run ruff check --fix src tests

# Check specific file
uv run ruff check src/app.py
```

### Running the Formatter

```bash
# Check formatting (dry run, no changes)
uv run ruff format --check src tests

# Format all files
uv run ruff format src tests

# Format specific file
uv run ruff format src/app.py
```

### Common Workflow

```bash
# 1. Check for issues
uv run ruff check src tests

# 2. Fix auto-fixable issues
uv run ruff check --fix src tests

# 3. Format code
uv run ruff format src tests

# 4. Verify everything passes
uv run ruff check src tests
```

## CI Integration

The lint workflow runs on every PR and push to main:

```yaml
- name: Check formatting
  run: uv run ruff format --check src tests

- name: Check linting
  run: uv run ruff check src tests
```

## Common Issues and Fixes

### F841: Unused Variable

```python
# Bad
result = slow_function()  # result never used

# Good
slow_function()

# Or use underscore prefix
_result = slow_function()  # indicates intentionally unused
```

### F821: Undefined Name

```python
# Bad
return {"cuda_available": cuda_available}  # cuda_available not defined

# Good
return {"cuda_available": torch.cuda.is_available()}
```

### B904: Exception Chaining

```python
# Bad
except TimeoutError:
    raise HTTPException(status_code=504)

# Good
except TimeoutError as exc:
    raise HTTPException(status_code=504) from exc
```

### UP038: Modern isinstance

```python
# Old (still works but not preferred)
isinstance(x, (int, float))

# Modern Python 3.13+
isinstance(x, int | float)
```

### I001: Import Sorting

Ruff automatically sorts imports in this order:
1. Standard library
2. Third-party
3. First-party (your code)

```python
# Before (unsorted)
import torch
import os
from src.utils import helper
import numpy as np

# After (sorted)
import os

import numpy as np
import torch

from src.utils import helper
```

## Gradual Adoption

If introducing ruff to an existing codebase, you can start with a minimal rule set and gradually add more:

```toml
[tool.ruff.lint]
# Start minimal
select = ["F", "E"]

# Add more as codebase improves
select = ["F", "E", "I"]      # Add import sorting
select = ["F", "E", "I", "B"]  # Add bugbear
```

## Additional Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Ruff Rules Reference](https://docs.astral.sh/ruff/rules/)
- [Configuration Options](https://docs.astral.sh/ruff/configuration/)
