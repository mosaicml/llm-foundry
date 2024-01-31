# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG DEP_GROUPS

# Install and uninstall foundry to cache foundry requirements
RUN git clone -b main https://github.com/mosaicml/llm-foundry.git
RUN pip install --no-cache-dir --upgrade "./llm-foundry${DEP_GROUPS}"
RUN pip uninstall -y llm-foundry
RUN rm -rf llm-foundry
