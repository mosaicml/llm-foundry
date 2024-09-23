from llmfoundry.utils.builders import build_logger

from composer.loggers import MosaicMLLogger

def test_mosaic_ml_logger_constructs():
    mosaic_ml_logger = build_logger('mosaicml', kwargs={'ignore_exceptions': True})

    assert isinstance(mosaic_ml_logger, MosaicMLLogger)
    assert mosaic_ml_logger.ignore_exceptions == True