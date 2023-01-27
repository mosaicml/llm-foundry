#!/bin/env bash

# Script to cd into a subdirectory, install its deps, and lint its contents
#
# The one subtlety here is that we only install the CPU dependencies since
# we don't have the CI infra to run workflows on GPUs yet. Also, it makes it
# easy to run this on your local machine + checks that we're doing
# conditional imports properly.

ENV_NAME="env-${1%/}"   # strip trailing slash if present

echo "Creating venv..."
python -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

echo "Installing requirements..."
pip install --upgrade pip
pip install ".[$1-cpu]"  # setup.py merges repo + subdir deps + strips gpu deps

echo "Running checks on files:"
FILES=$(find "$1" -type f | grep -v '.pyc')
echo $FILES
pre-commit run --files $FILES && pyright $FILES
STATUS=$?

echo "Cleaning up venv..."
deactivate
rm -rf "$ENV_NAME"

exit $STATUS
