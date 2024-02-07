#!/bin/bash

# This script should be used as a template for any other script

# Help Function
show_help() {
  echo "This script should be used as a template for any other script"
  echo
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  -h, --help         Display this help message."
  echo "  -p, --path PATH    Specify the path for the script. (Required)"
}

# Parse command-line options
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) # Help
      show_help
      exit 0
      ;;
    -p|--path) # Provide Path
      shift
      PATH="$1"
      ;;
    *) # Unknown option
      echo "Unknown option: $1."
      show_help
      exit 1
      ;;
  esac
  shift
done