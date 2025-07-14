#!/usr/bin/env just --justfile

# Load from '.env' file
set dotenv-load

# List available commands
help:
    @just --justfile {{justfile()}} --list --unsorted

# Ruff
lint:
  # Python ruff lint
  ./scripts/ruff_lint.sh

# SQLFluff
lint_sql file="./queries":
  # SQL Fix and lint
  ./scripts/sqlfluff_fix_and_lint.sh {{file}}

# Run pre-commit
pre:
  pre-commit run --all-files

# Run Pytest
test:
  uv run pytest

# Launch Jupyter Lab
jupy:
  uv run jupyter lab