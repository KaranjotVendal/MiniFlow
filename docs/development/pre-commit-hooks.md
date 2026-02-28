# Pre-commit Hooks

This document describes the pre-commit hooks configured for MiniFlow and how to use them.

## Overview

[Pre-commit](https://pre-commit.com/) is a framework for managing and maintaining multi-language pre-commit hooks. It runs checks automatically before each commit, catching issues before they reach CI.

## Why Pre-commit?

- **Early Feedback**: Catch issues before committing
- **Consistent Style**: Enforce code style automatically
- **CI Savings**: Fewer failed CI runs due to style issues
- **Team Alignment**: Everyone uses the same checks

## Installation

### 1. Install Pre-commit (included in dev dependencies)

```bash
# If not already installed
uv sync --group dev
```

### 2. Install Git Hooks

```bash
# Install the pre-commit hook into your .git/hooks
uv run pre-commit install
```

You should see:
```
pre-commit installed at .git/hooks/pre-commit
```

### 3. (Optional) Install Hooks for All Repos

```bash
# If you want pre-commit in all your repos
uv run pre-commit init-templatedir ~/.git-template
```

## Configuration

Pre-commit is configured in `.pre-commit-config.yaml`:

```yaml
repos:
  # Ruff - linting and formatting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.0
    hooks:
      - id: ruff
        args: [--fix]
        files: ^(src|tests)/
      - id: ruff-format
        files: ^(src|tests)/

  # General file checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
        exclude: ^(\.venv/|vibevoice/)
      - id: end-of-file-fixer
        exclude: ^(\.venv/|vibevoice/)
      - id: check-merge-conflict
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: debug-statements
        files: ^(src|tests)/
      - id: check-yaml
      - id: check-json
      - id: no-commit-to-branch
        args: ['--branch', 'main', '--branch', 'master']
```

## Available Hooks

### Ruff Hooks

| Hook | Purpose | When It Runs |
|------|---------|--------------|
| `ruff` | Lint Python files with auto-fix | On commit |
| `ruff-format` | Format Python files | On commit |

### General Hooks

| Hook | Purpose |
|------|---------|
| `trailing-whitespace` | Remove trailing whitespace |
| `end-of-file-fixer` | Ensure files end with newline |
| `check-merge-conflict` | Detect merge conflict markers (`<<<<<<< HEAD`) |
| `check-added-large-files` | Block files larger than 1MB |
| `debug-statements` | Catch `breakpoint()`, `pdb`, `ipdb` |
| `check-yaml` | Validate YAML syntax |
| `check-json` | Validate JSON syntax |
| `no-commit-to-branch` | Block direct commits to main/master |

## Usage

### Normal Workflow (Automatic)

Hooks run automatically on every commit:

```bash
git add .
git commit -m "my feature"
# Hooks run here...
```

If hooks pass:
```
ruff.....................................................................Passed
ruff-format..............................................................Passed
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
...
[main 1234567] my feature
```

If hooks fail:
```
ruff.....................................................................Failed
- hook id: ruff
- exit code: 1
src/app.py:10:5: F841 Local variable `x` is assigned to but never used
```

Fix the issues and commit again:
```bash
# Fix the issues
uv run ruff check --fix src

# Or manually edit the files
# Then commit again
git add .
git commit -m "my feature"
```

### Manual Usage

```bash
# Run all hooks on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff --all-files

# Run on specific file
uv run pre-commit run --files src/app.py

# Run without installing (one-time)
uv run pre-commit run --all-files --no-install
```

### Skip Hooks (Emergency)

```bash
# Skip pre-commit hooks for this commit only
# ⚠️ Use sparingly - CI will still run checks
git commit -m "hotfix" --no-verify
```

## Updating Hooks

```bash
# Update to latest versions
uv run pre-commit autoupdate

# Update specific repo
uv run pre-commit autoupdate --repo https://github.com/astral-sh/ruff-pre-commit
```

## CI Integration

Pre-commit hooks complement CI but don't replace it. The CI workflow runs the same checks:

```yaml
# .github/workflows/lint.yml
- name: Check formatting
  run: uv run ruff format --check src tests

- name: Check linting
  run: uv run ruff check src tests
```

## Troubleshooting

### Hook Installation Fails

```bash
# Clean and reinstall
uv run pre-commit clean
uv run pre-commit install
```

### Hook Takes Too Long

```bash
# Skip specific slow hooks
SKIP=ruff-format git commit -m "quick fix"
```

### Hook Environment Issues

```bash
# Clean all environments
uv run pre-commit clean

# Reinstall
uv run pre-commit install
```

### Bypass Pre-commit Temporarily

```bash
# Commit without hooks (not recommended)
git commit -m "message" --no-verify
```

## Best Practices

1. **Install Early**: Install pre-commit hooks right after cloning the repo
2. **Don't Skip**: Only use `--no-verify` in emergencies
3. **Fix Issues**: Address hook failures before pushing
4. **Stay Updated**: Run `autoupdate` periodically
5. **Team Alignment**: Ensure all team members have hooks installed

## Alternative: Manual Checking

If you prefer not to use pre-commit hooks, run the same checks manually:

```bash
# Format code
uv run ruff format src tests

# Check linting
uv run ruff check src tests

# Fix auto-fixable issues
uv run ruff check --fix src tests
```

However, we strongly recommend using pre-commit hooks to catch issues early.
