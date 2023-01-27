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
python -m venv "$ENV_NAME" --system-site-packages
source "$ENV_NAME/bin/activate"

echo "Installing requirements..."
pip install --upgrade pip
pip install ".[$1-cpu]"  # we rely on docker image to handle flash-attn, etc

cp pyproject.toml "$1"
cd "$1"

# run tests using project pytest config
python -m pytest tests
rm pyproject.toml

echo "Cleaning up venv..."
deactivate

cd -
rm -rf "$ENV_NAME"
