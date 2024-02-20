#!/usr/bin/env just --justfile

# Load from '.env' file
set dotenv-load

# List available commands
help:
    @just --justfile {{justfile()}} --list --unsorted

# PyLint
lint:
  # PyLint lint from ./src
  ./scripts/pylint_lint.sh

# SQLFluff
lint_sql file="./queries":
  # Fix and lint
  ./scripts/sqlfluff_fix_and_lint.sh {{file}}

# Start Jupyter Lab
jupy:
  poetry run jupyter lab

# Test .env file
test_env_file:
  echo $TEST_ENV_VAR