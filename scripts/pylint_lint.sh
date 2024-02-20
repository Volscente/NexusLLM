#!/bin/bash
#
# Run PyLint with specific plugins loaded and message template

echo "-------- PyLint Lint --------"
echo

# Lint
poetry run pylint --load-plugins pylint_pytest \
  --source-roots=./src \
  --output-format=colorized \
  --msg-template='Rule: {msg_id} - Position: [{line},{column}] -  {msg}' \
  ./src ./tests

echo "--------------------------------"
echo