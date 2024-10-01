# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG BRANCH_NAME
ARG DEP_GROUPS
ARG TE_COMMIT
ARG KEEP_FOUNDRY=false

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 8.9 9.0"

# Check for changes in setup.py.
# If there are changes, the docker cache is invalidated and a fresh pip installation is triggered.
ADD https://raw.githubusercontent.com/mosaicml/llm-foundry/$BRANCH_NAME/setup.py setup.py

# Install TransformerEngine
RUN NVTE_FRAMEWORK=pytorch CMAKE_BUILD_PARALLEL_LEVEL=4 MAX_JOBS=4 pip install git+https://github.com/NVIDIA/TransformerEngine.git@$TE_COMMIT

# Install and uninstall foundry to cache foundry requirements
RUN echo "Cloning llm-foundry repo from branch $BRANCH_NAME..." && \
    git clone -b $BRANCH_NAME https://github.com/mosaicml/llm-foundry.git /llm-foundry && \
    echo "Repository cloned successfully into /llm-foundry" && \
    echo "Current directory after cloning: $(pwd)" && \
    echo "Listing contents of /llm-foundry to confirm cloning:" && \
    ls -al /llm-foundry && \
    echo "Installing llm-foundry with dependency groups: $DEP_GROUPS" && \
    pip install --no-cache-dir "/llm-foundry${DEP_GROUPS}" && \
    echo "llm-foundry installation complete."

# Conditionally uninstall llm-foundry and remove its directory
RUN if [ "$KEEP_FOUNDRY" != "true" ]; then \
      echo "Uninstalling llm-foundry..." && \
      pip uninstall -y llm-foundry && \
      echo "llm-foundry uninstalled." && \
      echo "Removing /llm-foundry directory..." && \
      rm -rf /llm-foundry && \
      echo "Directory /llm-foundry removed." && \
    else \
      echo "KEEP_FOUNDRY is set to true, not uninstalling llm-foundry." && \
    fi

# Final directory check
RUN echo "Current directory at end of build: $(pwd)"
