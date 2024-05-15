# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG BRANCH_NAME
ARG DEP_GROUPS

# Check for changes in setup.py.
# If there are changes, the docker cache is invalidated and a fresh pip installation is triggered.
ADD https://raw.githubusercontent.com/mosaicml/llm-foundry/$BRANCH_NAME/setup.py setup.py
RUN rm setup.py

# Install and uninstall foundry to cache foundry requirements
RUN git clone -b $BRANCH_NAME https://github.com/mosaicml/llm-foundry.git && \
    pip install --no-cache-dir "./llm-foundry${DEP_GROUPS}" && \
    pip uninstall -y llm-foundry && \
    rm -rf llm-foundry && export NVTE_FRAMEWORK=pytorch && MAX_JOBS=1 pip install --verbose git+https://github.com/NVIDIA/TransformerEngine.git@main
