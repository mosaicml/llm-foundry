# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG BRANCH_NAME
ARG DEP_GROUPS
ARG KEEP_FOUNDRY=false

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 8.9 9.0"

# Check for changes in setup.py.
# If there are changes, the docker cache is invalidated and a fresh pip installation is triggered.
ADD https://raw.githubusercontent.com/mosaicml/llm-foundry/$BRANCH_NAME/setup.py setup.py
RUN rm setup.py

# Install and uninstall foundry to cache foundry requirements
RUN git clone -b $BRANCH_NAME https://github.com/mosaicml/llm-foundry.git
RUN pip install --no-cache-dir "./llm-foundry${DEP_GROUPS}"

# Conditionally uninstall llm-foundry and remove its directory
RUN if [ "$KEEP_FOUNDRY" != "true" ]; then \
      pip uninstall -y llm-foundry && \
      rm -rf /llm-foundry; \
    fi
