#!/bin/env bash

# Script to cd into a subdirectory, install its deps, and run its tests
#
# The tricky part here is that some modules, like flash-attn, are really
# finicky to get installed and depend on details of your environment (e.g.,
# the installed CUDA version). For these deps, we just use whatever's
# in the surrounding environment. We do this by:
#   1) Allowing the venv to use the environment's installed packages
#       via the --system-site-packages flag.
#   2) Having the pip install overwrite everything in the requirements
#       *except* a few whitelisted dependencies.

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
