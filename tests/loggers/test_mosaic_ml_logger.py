# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.loggers import MosaicMLLogger

from llmfoundry.utils.builders import build_logger


def test_mosaic_ml_logger_constructs():
    mosaic_ml_logger = build_logger(
        'mosaicml',
        kwargs={'ignore_exceptions': True},
    )

    assert isinstance(mosaic_ml_logger, MosaicMLLogger)
    assert mosaic_ml_logger.ignore_exceptions == True
