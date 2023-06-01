# CCopyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY setup.py setup.py
COPY llmfoundry llmfoundry
RUN pip install --no-cache-dir ".[gpu]" && \
    pip uninstall -y llmfoundry && \
    rm setup.py && \
    rm -rf llmfoundry
