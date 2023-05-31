# CCopyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY setup.py setup.py
COPY __init__.py __init__.py
RUN pip install --no-cache-dir ".[gpu]" && \
    rm __init__.py setup.py
