# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
FROM $BASE_IMAGE


# Install and uninstall foundry to cache foundry requirements
RUN git clone -b main https://github.com/mosaicml/llm-foundry.git && \
    pip install --no-cache-dir "./llm-foundry[gpu]" && \
    pip uninstall -y llm-foundry && \
    rm -rf llm-foundry
