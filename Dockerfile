# CCopyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
FROM $BASE_IMAGE


RUN git clone -b main https://github.com/mosaicml/llm-foundry.git && \
    pip install --no-cache-dir "./llm-foundry[gpu]" && \
    pip uninstall -y llmfoundry && \
    rm -rf llm-foundry
