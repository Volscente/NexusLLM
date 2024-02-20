#!/bin/bash
#
# First fix and then lint BigQuery SQL Queries

# Check for the presence of argument file
if [ $# -ne 1 ]; then
  echo "Usage: $0 <file>"
  exit 1
fi

# Assign path to fix & lint
file="$1"

echo "-------- SQLFluff Fix & Lint --------"
echo

# Fix & lint
poetry run sqlfluff fix --dialect bigquery --exclude-rules LT05 "$file" && poetry run sqlfluff lint --dialect bigquery --exclude-rules LT05 "$file"

echo "----------------------------------------"
echo