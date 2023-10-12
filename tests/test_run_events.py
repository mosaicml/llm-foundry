from unittest.mock import patch

import pytest
from composer.loggers.mosaicml_logger import (MOSAICML_ACCESS_TOKEN_ENV_VAR,
                                              MOSAICML_PLATFORM_ENV_VAR,
                                              MosaicMLLogger,)

from llmfoundry.callbacks import RunEventsCallback
from llmfoundry.utils.builders import build_logger


@pytest.mark.parametrize('platform_env_var', ['True', None])
@pytest.mark.parametrize('access_token_env_var', ['my-token', None])
def testMosaicLogger(monkeypatch: pytest.MonkeyPatch, platform_env_var: str, access_token_env_var: str):
    if platform_env_var:
        monkeypatch.setenv(MOSAICML_PLATFORM_ENV_VAR, platform_env_var)
    if access_token_env_var:
        monkeypatch.setenv(MOSAICML_ACCESS_TOKEN_ENV_VAR, access_token_env_var)
    monkeypatch.setenv('RUN_NAME', 'test-run-name')

    if platform_env_var == 'True' and access_token_env_var == 'my-token':
        destination = build_logger('mosaicml', {})
        assert destination is not None
    else:
        with pytest.raises(ValueError, match="Not sure how to build logger: mosaicml"):
            build_logger('mosaicml', {})
   
def testRunEvents(monkeypatch: pytest.MonkeyPatch): 
    with patch.object(MosaicMLLogger, 'log_metrics', autospec=True) as log_metrics:
        monkeypatch.setenv('RUN_NAME', 'test-run-name')
        RunEventsCallback().data_validated(MosaicMLLogger(), 1000)
        
        log_metrics.assert_called_once()
        args, _ = log_metrics.call_args
        metrics = args[1]
        assert isinstance(metrics, dict)
        assert 'data_validated' in metrics
        assert 'total_num_samples' in metrics
        assert isinstance(metrics['data_validated'], float)
        assert isinstance(metrics['total_num_samples'], int)
