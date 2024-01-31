# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG DEP_GROUPS
ARG BRANCH_NAME

# Install and uninstall foundry to cache foundry requirements
ADD https://raw.githubusercontent.com/mosaicml/llm-foundry/$BRANCH_NAME/setup.py setup.py
RUN git clone -b $BRANCH_NAME https://github.com/mosaicml/llm-foundry.git
RUN pip install --no-cache-dir "./llm-foundry${DEP_GROUPS}"
RUN pip uninstall -y llm-foundry
RUN rm -rf llm-foundry
RUN rm setup.py

RUN pip freeze
