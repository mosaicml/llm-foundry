# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

FROM mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04

ARG EXAMPLE
COPY ${EXAMPLE}/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt
