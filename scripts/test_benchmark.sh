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

ENV_NAME="${1%/}-env"   # strip trailing slash if present

cd "$1"

echo "Creating venv..."
python -m venv "$ENV_NAME" --system-site-packages
source "$ENV_NAME/bin/activate"

cat requirements.txt | grep -v 'flash-attn' > /tmp/requirements.txt

echo "Installing requirements:"
cat /tmp/requirements.txt
# TODO -I would sandbox better (always overwriting system copies with versions
# in requirements.txt) but this causes mysterious Flash Attention issues
pip install -U -r /tmp/requirements.txt
# We need to force install pytest into each environment so that it does not use the system pytest
pip install --ignore-installed pytest
rm /tmp/requirements.txt
python -m pytest tests

echo "Cleaning up venv..."
deactivate
rm -rf "$ENV_NAME"

cd -
